# References:
# https://jatinbansal.com/ai-engineering/rag-evaluation/

import json
from iv_bm25_retrieval import retrieve_from_bm25
from iv_dense_retrieval import load_embedding_model, dense_retrieve
import json
from datasets import load_from_disk
import re

paper_ids = [2146, 10884, 13995, 17302, 19577, 21665, 24216, 25475, 899, 1625]
custom_corpus = load_from_disk("custom_corpus")
documents = []
for paper in custom_corpus:
  no_cite = re.sub(r'(?<=\D)\.(\d+)', '.', paper["article"])
  normalize_whitespace = re.sub(r"\s+", " ", no_cite).strip()
  documents.append(normalize_whitespace)

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
  # The number of gold documents is the number of excerpts in the correct answer (those separated by ...)
  # Recall is calculated as the number of gold documents that are included in the retrieved documents, and partially overlapping chunks count
  for k in 5, 10, 50:
    for method in chunking_methods:
      # Calculating the average recall over all questions
      recall_sum = 0
      for question_record in evaluation_questions:
        correct_locations = question_record["answer_indices"]
        query = question_record["question"]
        top_k = retrieve_from_bm25(query, method, n=k)
        num_found = 0
        for answer_start, answer_end in correct_locations:
          found = False
          for chunk in top_k:
            if chunk["start"] < answer_end and chunk["end"] > answer_start:
              found = True
              break
          if found:
            num_found += 1
        recall = num_found / len(correct_locations)
        recall_sum += recall
      avg_recall = recall_sum/len(evaluation_questions)
      print("Chunking method:", method, "Recall@" + str(k) + " =", avg_recall)

    print("Dense Retrieval:")
    for k in 5, 10, 50:
      for chunking_method in chunking_methods:
        for embedding_model in ["OpenAI", "BioBERT", "BGE", "MedCPT"]:
          # Loads the selected embedding model once and reuses it for all queries.
          model_resources = load_embedding_model(embedding_model=embedding_model)
          # Calculating the average recall over all questions
          recall_sum = 0
          for question_record in evaluation_questions:
            correct_locations = question_record["answer_indices"]
            query = question_record["question"]
            results = dense_retrieve(
              query=query,
              embedding_model=embedding_model,
              chunking_method=chunking_method,
              model_resources=model_resources,
              top_k=k
            )
            num_found = 0
            for answer_start, answer_end in correct_locations:
              found = False
              for chunk in results:
                if chunk["start"] < answer_end and chunk["end"] > answer_start:
                  found = True
                  break
              if found:
                num_found += 1
            recall = num_found / len(correct_locations)
            recall_sum += recall
          avg_recall = recall_sum/len(evaluation_questions)
          print("Chunking method:", method, "- Embedding model:", embedding_model, "- Recall@" + str(k) + " =", avg_recall)