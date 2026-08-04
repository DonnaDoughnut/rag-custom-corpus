import json
from datasets import load_from_disk
import re

# custom_corpus = load_from_disk("custom_corpus")
# documents = []
# for paper in custom_corpus:
#     documents.append(re.sub(r'(?<=\D)\.(\d+)', '.', paper["article"]))  # Remove most citation numbers, because they confuse the sentence splitter

# with open("evaluation_questions.txt", "r", encoding="utf-8") as file:
#   evaluation_questions = json.load(file)
# for question in evaluation_questions:
#   answer_chunks = question["answer_span"].split("...")
#   paper_id = question["paper_id"]
#   for chunk in answer_chunks:
    

# new_version = open("eval_qs_with_indices.txt", "w")