# ======================================================================
# This file evaluates 
#   1. BM25 + cross-encoder, 
#   2. dense + cross-encoder, and
#   3. hybrid ranking 
# using Recall@5, Recall@10, and Recall@50.
# ======================================================================


import json

from iv_bm25_retrieval import retrieve_from_bm25
from iv_dense_retrieval import dense_retrieve, load_embedding_model
from vi_cross_encoder_reranking import load_cross_encoder, cross_encoder_rerank
from vi_hybrid_ranking import hybrid_rerank


chunking_methods = [
    "fixed_token",
    "recursive",
    "sentence",
    "semantic"
]

embedding_models = [
    "OpenAI",
    "BioBERT",
    "BGE",
    "MedCPT"
]

candidate_k = 100
evaluation_ks = [5, 10, 50]


# 1. Loads the evaluation questions.
with open("eval_qs_with_indices.txt", "r", encoding="utf-8") as file:
    evaluation_questions = json.load(file)


# 2. Calculates recall at k. (Same method used in v_evaluation.py)
def calculate_recall(results, correct_locations, k):

    num_found = 0

    for answer_start, answer_end in correct_locations:

        for chunk in results[:k]:
            if chunk["start"] < answer_end and chunk["end"] > answer_start:
                num_found += 1
                break

    return num_found / len(correct_locations)


if __name__ == "__main__":

    # Opens a txt file to store evaluation results.
    output_file = open(
        "vii_evaluation2_results.txt",
        "w",
        encoding="utf-8"
    )

    # Prints results to both terminal and txt file.
    def log(*args):
        print(*args)
        print(*args, file=output_file)


    # Loads the cross-encoder model.
    cross_encoder_model = load_cross_encoder(
        "cross-encoder/ms-marco-MiniLM-L6-v2"
    )


    # 3. Evaluates BM25 + cross-encoder
    log("BM25 + Cross-Encoder:")

    for chunking_method in chunking_methods:

        recall_sums = {5: 0, 10: 0, 50: 0}

        for question_record in evaluation_questions:

            query = question_record["question"]
            correct_locations = question_record["answer_indices"]

            # Gets the top 100 BM25 candidates.
            bm25_results = retrieve_from_bm25(
                query=query,
                method=chunking_method,
                n=candidate_k
            )

            # Reranks BM25 results using cross-encoder.
            bm25_ce_results = cross_encoder_rerank(
                query=query,
                retrieved_docs=bm25_results,
                cross_encoder_model=cross_encoder_model,
                top_k=50
            )

            for k in evaluation_ks:

                recall_sums[k] += calculate_recall(
                    bm25_ce_results,
                    correct_locations,
                    k
                )

        for k in evaluation_ks:

            average_recall = (
                recall_sums[k]
                / len(evaluation_questions)
            )

            log(
                "Chunking method:",
                chunking_method,
                "Recall@" + str(k) + " =",
                average_recall
            )


    # 4. Evaluates Dense + cross-encoder
    log("Dense + Cross-Encoder:")

    for chunking_method in chunking_methods:

        for embedding_model in embedding_models:

            model_resources = load_embedding_model(
                embedding_model=embedding_model
            )

            recall_sums = {5: 0, 10: 0, 50: 0}

            for question_record in evaluation_questions:

                query = question_record["question"]
                correct_locations = question_record["answer_indices"]

                # Gets the top 100 dense retrieval candidates.
                dense_results = dense_retrieve(
                    query=query,
                    embedding_model=embedding_model,
                    chunking_method=chunking_method,
                    model_resources=model_resources,
                    top_k=candidate_k
                )

                # Reranks dense results using cross-encoder.
                dense_ce_results = cross_encoder_rerank(
                    query=query,
                    retrieved_docs=dense_results,
                    cross_encoder_model=cross_encoder_model,
                    top_k=50
                )

                for k in evaluation_ks:

                    recall_sums[k] += calculate_recall(
                        dense_ce_results,
                        correct_locations,
                        k
                    )

            for k in evaluation_ks:

                average_recall = (
                    recall_sums[k]
                    / len(evaluation_questions)
                )

                log(
                    "Chunking method:",
                    chunking_method,
                    "- Embedding model:",
                    embedding_model,
                    "- Recall@" + str(k) + " =",
                    average_recall
                )


    # 5. Evaluates hybrid ranking
    log("Hybrid Ranking:")

    for chunking_method in chunking_methods:

        for embedding_model in embedding_models:

            model_resources = load_embedding_model(
                embedding_model=embedding_model
            )

            recall_sums = {5: 0, 10: 0, 50: 0}

            for question_record in evaluation_questions:

                query = question_record["question"]
                correct_locations = question_record["answer_indices"]

                # Combines BM25 and dense retrieval using hybrid ranking.
                hybrid_results = hybrid_rerank(
                    query=query,
                    chunking_method=chunking_method,
                    embedding_model=embedding_model,
                    model_resources=model_resources,
                    candidate_k=candidate_k,
                    top_k=50
                )

                for k in evaluation_ks:

                    recall_sums[k] += calculate_recall(
                        hybrid_results,
                        correct_locations,
                        k
                    )

            for k in evaluation_ks:

                average_recall = (
                    recall_sums[k]
                    / len(evaluation_questions)
                )

                log(
                    "Chunking method:",
                    chunking_method,
                    "- Embedding model:",
                    embedding_model,
                    "- Recall@" + str(k) + " =",
                    average_recall
                )


    output_file.close()