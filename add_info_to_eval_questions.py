from datasets import load_from_disk
from datasets import load_dataset
import re
import json
import sys
import pickle
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize

dataset = load_dataset(
    "armanc/scientific_papers",
    "pubmed",
    split="train",
    trust_remote_code=True
)
custom_corpus = load_from_disk("custom_corpus")

# Identifying the duplicate sentences in each paper.
with open("bm25_indexes.pkl", "rb") as file:
  loaded_bm25_data = pickle.load(file)
chunks = loaded_bm25_data["sentence"]["chunks"]
prev_paper = None
sentences = []
for chunk in chunks:
  if chunk["paper_id"] != prev_paper and prev_paper != None:
    print("Paper:", prev_paper)
    counts = Counter(sentences)
    # for sentence, count in counts.items():
    #   if count > 1:
    #     print(f"Repeated {count} times:")
    #     print(sentence)
    #     print()
    duplicate_sentences = sum(count - 1 for count in counts.values() if count > 1)
    print(f"{duplicate_sentences} duplicate sentences out of {len(sentences)} sentences. This means {100 * duplicate_sentences / len(sentences):.1f}% duplicates.")
    sentences = []

  sentences.extend(s.strip() for s in sent_tokenize(chunk["text"]))
  prev_paper = chunk["paper_id"]

print("Paper:", prev_paper)
counts = Counter(sentences)
# for sentence, count in counts.items():
#   if count > 1:
#     print(f"Repeated {count} times:")
#     print(sentence)
#     print()
duplicate_sentences = sum(count - 1 for count in counts.values() if count > 1)
print(f"{duplicate_sentences} duplicate sentences out of {len(sentences)} sentences. This means {100 * duplicate_sentences / len(sentences):.1f}% duplicates.")

sys.exit()



# Tokenize the query
tokenized_query = tokenize_for_bm25(query)

# Retrieve documents using BM25
retrieved_docs = bm25_index.get_top_n(tokenized_query, chunks, n)

article_fulltexts = []
for article in custom_corpus:
  article_fulltexts.append(re.sub(r'(?<=\D)\.(\d+)', '.', article["article"]))  # Remove most citation numbers, because they confuse the sentence splitter

with open("evaluation_questions.txt", "r", encoding="utf-8") as file:
  evaluation_questions = json.load(file)
# for question in evaluation_questions:

new_version = open("eval_qs_with_indices.txt", "w")