from datasets import load_from_disk
from datasets import load_dataset
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
