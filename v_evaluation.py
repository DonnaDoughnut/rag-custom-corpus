# References:
# https://jatinbansal.com/ai-engineering/rag-evaluation/

import json
from v_bm25_retrieval import retrieve_from_bm25
from v_dense_retrieval import load_embedding_model, dense_retrieve

chunking_methods = [
  "fixed_token",
  "recursive",
  "sentence",
  "semantic"
]

# Loads all evaluation questions from the JSON-formatted `evaluation_questions.txt`.
with open("evaluation_questions.txt", "r", encoding="utf-8") as file:
    evaluation_questions = json.load(file)

if __name__ == "__main__":
  print("BM25:")
  # Need to evaluate at several levels (recall@5, recall@10, recall@50)
  for k in 5, 10, 50:
    for method in chunking_methods:
      num_bm25_successes = 0
      for question_record in evaluation_questions:
        query = question_record["question"]
        top_k = retrieve_from_bm25(query, method, n=k)
        found = False
        # Each question in the evaluation dataset has a chunk defined as the correct context, this is the paper that the chunk is in
        correct_paper = question_record["paper_id"]
        for doc in top_k:
          # Each doc is dict_keys(['text', 'document_id', 'paper_id', 'chunk_id', 'method'])
          if doc["paper_id"] == correct_paper:
            found = True
        if found:
          num_bm25_successes += 1

        # Testing
        import sys
        print("Chunking method:", method)
        print("Query:", query)
        print("Correct paper ID:", correct_paper)
        for doc in top_k:
          print("Paper", doc["paper_id"], "chunk", doc["chunk_id"], "retrieved")
          print("Text:", doc["text"])
        sys.exit()

      print("Recall@" + str(k), "- Chunking method:", method, "- Number of questions for which it found the right paper context:",
            num_bm25_successes, "out of", len(evaluation_questions), "questions")

    print("Dense Retrieval:")
    for k in 5, 10, 50:
      for chunking_method in chunking_methods:
        for embedding_model in ["OpenAI", "BioBERT", "BGE", "MedCPT"]:
          # Loads the selected embedding model once and reuses it for all queries.
          model_resources = load_embedding_model(embedding_model=embedding_model)
          num_dense_successes = 0
          for question_record in evaluation_questions:
            query = question_record["question"]
            results = dense_retrieve(
              query=query,
              embedding_model=embedding_model,
              chunking_method=chunking_method,
              model_resources=model_resources,
              top_k=k
            )
            found = False
            correct_paper = question_record["paper_id"]
            for doc in results:
              if doc["paper_id"] == correct_paper:
                found = True
            if found:
              num_dense_successes += 1
          print("Recall@" + str(k), "- Chunking method:", chunking_method, "- Embedding model:", embedding_model,
                "- Number of questions for which it found the right paper context:",
                num_dense_successes, "out of", len(evaluation_questions), "questions")