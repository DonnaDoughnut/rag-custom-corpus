import json
from datasets import load_from_disk
import re

paper_ids = [2146, 10884, 13995, 17302, 19577, 21665, 24216, 25475, 899, 1625]
custom_corpus = load_from_disk("custom_corpus")
documents = []
for paper in custom_corpus:
    documents.append(re.sub(r'(?<=\D)\.(\d+)', '.', paper["article"]))  # Remove most citation numbers, because they confuse the sentence splitter

with open("evaluation_questions.txt", "r", encoding="utf-8") as file:
  evaluation_questions = json.load(file)
new_eval_json = []  # List of jsons
for question in evaluation_questions:
  answer_chunks = question["answer_span"].split("...")
  paper_id = question["paper_id"]
  paper_text = documents[paper_ids.index(paper_id)]
  chunk_locs = []  # Lists of length 2 of start-end indices
  for chunk in answer_chunks:
    if chunk in paper_text:
      chunk_start = paper_text.index(chunk)
      chunk_end = chunk_start + len(chunk)
      chunk_locs.append([chunk_start, chunk_end])
    else:
      print("Chunk not found in paper.")
      print("Chunk:", chunk)
      print("Full text:", paper_text)

  question["answer_indices"] = chunk_locs
  new_eval_json.append(question)

with open("eval_qs_with_indices.txt", "w", encoding="utf-8") as f:
    json.dump(new_eval_json, f, indent=2, ensure_ascii=False)