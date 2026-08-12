import json

from iv_bm25_retrieval import retrieve_from_bm25
from iv_dense_retrieval import dense_retrieve, load_embedding_model


# 1. Helper: Uses paper ID and chunk ID to identify the same chunk.
def get_chunk_key(doc):
    return (
        doc["paper_id"],
        doc["chunk_id"]
    )


# 2. Combines the BM25 and dense retrieval rankings using RRF
#    and returns the top-k results.
def reciprocal_rank_fusion(
    bm25_results,
    dense_results,
    top_k=5,
    rrf_k=60,
    bm25_weight=1.0,
    dense_weight=1.0
):

    if top_k <= 0:
        raise ValueError("Error: top_k must be greater than zero.")

    if rrf_k < 0:
        raise ValueError("Error: rrf_k cannot be negative.")

    fused_candidates = {}

    # Adds the BM25 rank and RRF score.
    for bm25_rank, doc in enumerate(bm25_results, start=1):
        key = get_chunk_key(doc)

        if key not in fused_candidates:
            fused_candidates[key] = {
                "doc": dict(doc),
                "bm25_rank": None,
                "dense_rank": None,
                "dense_similarity_score": None,
                "hybrid_score": 0.0
            }

        fused_candidates[key]["bm25_rank"] = bm25_rank

        fused_candidates[key]["hybrid_score"] += (
            bm25_weight / (rrf_k + bm25_rank)
        )

    # Adds the dense rank and RRF score.
    for dense_position, doc in enumerate(dense_results, start=1):
        key = get_chunk_key(doc)

        if key not in fused_candidates:
            fused_candidates[key] = {
                "doc": dict(doc),
                "bm25_rank": None,
                "dense_rank": None,
                "dense_similarity_score": None,
                "hybrid_score": 0.0
            }

        dense_rank = doc.get("rank", dense_position)

        fused_candidates[key]["dense_rank"] = int(dense_rank)
        fused_candidates[key]["dense_similarity_score"] = doc.get(
            "similarity_score"
        )

        fused_candidates[key]["hybrid_score"] += (
            dense_weight / (rrf_k + dense_rank)
        )

        # Uses the dense version because it contains more retrieval information.
        fused_candidates[key]["doc"] = dict(doc)

    combined_results = []

    # Stores the BM25 rank, dense rank, and hybrid score for each candidate.
    for candidate in fused_candidates.values():
        result = dict(candidate["doc"])

        result["bm25_rank"] = candidate["bm25_rank"]
        result["dense_rank"] = candidate["dense_rank"]
        result["dense_similarity_score"] = candidate[
            "dense_similarity_score"
        ]
        result["hybrid_score"] = float(candidate["hybrid_score"])

        combined_results.append(result)

    combined_results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    top_k = min(top_k, len(combined_results))
    hybrid_results = combined_results[:top_k]

    for new_rank, result in enumerate(hybrid_results, start=1):
        result["rank"] = new_rank

    return hybrid_results


# 3. Runs BM25 and dense retrieval and combines their rankings using RRF.
def hybrid_rerank(
    query,
    chunking_method,
    embedding_model,
    model_resources,
    candidate_k=20,
    top_k=5,
    rrf_k=60,
    bm25_weight=1.0,
    dense_weight=1.0
):

    if candidate_k <= 0:
        raise ValueError("Error: candidate_k must be greater than zero.")

    bm25_results = retrieve_from_bm25(
        query=query,
        method=chunking_method,
        n=candidate_k
    )

    dense_results = dense_retrieve(
        query=query,
        embedding_model=embedding_model,
        chunking_method=chunking_method,
        model_resources=model_resources,
        top_k=candidate_k
    )

    bm25_keys = {get_chunk_key(doc) for doc in bm25_results}
    dense_keys = {get_chunk_key(doc) for doc in dense_results}

    common_keys = bm25_keys.intersection(dense_keys)

    print("BM25 candidates: ", len(bm25_keys))
    print("Dense candidates: ", len(dense_keys))
    print("Common candidates: ", len(common_keys))

    results = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
        top_k=top_k,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight
    )

    return results


# 4. Testing
if __name__ == "__main__":

    embedding_model = "BGE"
    chunking_method = "fixed_token"

    candidate_k = 20
    final_top_k = 5

    rrf_k = 60
    bm25_weight = 1.0
    dense_weight = 1.0

    dense_model_resources = load_embedding_model(
        embedding_model=embedding_model
    )

    with open("eval_qs_with_indices.txt", "r", encoding="utf-8") as file:
        evaluation_questions = json.load(file)

    # Only tests one query.
    question_record = evaluation_questions[2]
    query = question_record["question"]

    results = hybrid_rerank(
        query=query,
        chunking_method=chunking_method,
        embedding_model=embedding_model,
        model_resources=dense_model_resources,
        candidate_k=candidate_k,
        top_k=final_top_k,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight
    )

    print("Query: ", query)
    print()

    for result in results:
        print("=" * 60)
        print("BM25 rank: ", result["bm25_rank"])
        print("Dense rank: ", result["dense_rank"])
        print("New rank: ", result["rank"])
        print("Hybrid score: ", result["hybrid_score"])

        if result.get("dense_similarity_score") is not None:
            print(
                "Dense similarity score: ",
                result["dense_similarity_score"]
            )

        print("Text: ", result["text"])