import json
import numpy as np

from sentence_transformers import CrossEncoder
from iv_bm25_retrieval import retrieve_from_bm25
from iv_dense_retrieval import dense_retrieve, load_embedding_model

# 1. Loads the cross-encoder model
def load_cross_encoder(model_name="cross-encoder/ms-marco/MiniLM-L6-v2"):
    return CrossEncoder(model_name)

# 2. Reranks the retrieved chunks using the above cross-encoder 
#    and Returns the top-k most relevant results.
def cross_encoder_rerank(query, retrieved_docs, cross_encoder_model, top_k=5):

    if top_k <= 0:
        raise ValueError("Error: top_k must be greater than zero.")
    
    if not retrieved_docs:
        return []
    
    # Builds one (query, chunk) pair for each retrieved candidate.
    query_chunk_pairs = [(query, doc["text"]) for doc in retrieved_docs]

    # Scores all query-chunk pairs with the cross-encoder.
    cross_encoder_scores = cross_encoder_model.predict(query_chunk_pairs, show_progress_bar=False)

    cross_encoder_scores = np.array(
        cross_encoder_scores,
        dtype=np.float32
    ).reshape(-1)

    if len(cross_encoder_scores) != len(retrieved_docs):
        raise ValueError("Error: The # of cross-encoder scores need to match the # of retrieved docs.")
    
    # Stores each cadidate with its old retrieval rank and its new cross-encoder score.
    scored_candidates = []

    for retrieval_position, (doc, score) in enumerate(
        zip(retrieved_docs, cross_encoder_scores),
        start=1
    ):
        result = dict(doc)

        # Retrieves its previous dense retrieval rank as its old retrieval rank.
        # (BM25 doesn't have a rank.)
        original_rank = doc.get("rank", retrieval_position)
        result["retrieval_rank"] = int(original_rank)
        result["cross_encoder_score"] = float(score)

        result["retrieval_score"] = doc.get("similarity_score")

        scored_candidates.append(result)
    
    scored_candidates.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

    top_k = min(top_k, len(scored_candidates))
    reranked_results = scored_candidates[:top_k]

    for new_rank, result in enumerate(reranked_results, start=1):
        result["rank"] = new_rank

    return reranked_results


# 3. Testing
if __name__ == "__main__":

    cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
    cross_encoder_model = load_cross_encoder(model_name=cross_encoder_model_name)

    retrieval_type = "dense"
    chunking_method = "fixed_token"

    candidate_k = 20
    final_top_k = 5

    # ---------- For dense retrieval only ------------------
    embedding_model = "BGE"
    dense_model_resources = None
    if retrieval_type == "dense":
        dense_model_resources = load_embedding_model(
            embedding_model=embedding_model
        )
    # ------------------------------------------------------

    with open("eval_qs_with_indices.txt", "r", encoding="utf-8") as file:
        evaluation_questions = json.load(file)

    # Only tests the first query.
    question_record = evaluation_questions[0]
    query = question_record["question"]

    if retrieval_type == "dense":
        candidates = dense_retrieve(
            query=query,
            embedding_model=embedding_model,
            chunking_method=chunking_method,
            model_resources=dense_model_resources,
            top_k=candidate_k
        )
    
    elif retrieval_type == "bm25":
        candidates = retrieve_from_bm25(
            query=query,
            method=chunking_method,
            n=candidate_k
        )

    else:
        raise ValueError("Error: retrieval_type must be either dense or bm25.")
    
    # Cross-encoder ranking.
    results = cross_encoder_rerank(
        query=query,
        retrieved_docs=candidates,
        cross_encoder_model=cross_encoder_model,
        top_k=final_top_k
    )

    print("Query: ", query)
    print()

    for result in results:
        print("=" * 60)
        print("Old rank: ", result["retrieval_rank"])
        print("New rank: ", result["rank"])
        print("Cross-encoder score: ", result["cross_encoder_score"])
        print("Text: ", result["text"])
    


