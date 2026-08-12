"""Creates a dataset of 8 sleep papers and 2 exercise papers without any duplicated sentences."""

from datasets import load_dataset
import re

dataset = load_dataset(
    "armanc/scientific_papers",
    "pubmed",
    split="train",
    trust_remote_code=True
)

def search_papers_by_terms(dataset, term_groups, search_field, min_occurrences):
    
    if not term_groups or not term_groups[0]:
        raise ValueError("Error: At least one primary term group is required.")
    
    patterns = []

    for terms in term_groups:
        escaped_terms = [re.escape(term) for term in terms]

        pattern = re.compile(
            r"\b(?:" + "|".join(escaped_terms) + r")\b",
            re.IGNORECASE
        )
        patterns.append(pattern)
    
    matching_indices = []

    for index, paper in enumerate(dataset):
        text = paper.get(search_field, "") or ""

        primary_match = (len(patterns[0].findall(text)) >= min_occurrences)
        supporting_match = all(pattern.search(text) for pattern in patterns[1:])

        if primary_match and supporting_match:
            matching_indices.append(index)
    
    return matching_indices

# keywords = [["sleep disorder", "sleep quality", "insomnia"]]
keywords = [
    [
        "physical exercise",
        "physical activity",
        "exercise training",
        "walking exercise"
    ]
]

matching_indices = search_papers_by_terms(dataset, keywords, "abstract", 3)

print(f"Number of matching papers: {len(matching_indices)}")
print(f"Matching indices: {matching_indices[:20]}")

from ii_chunking_util import fixed_size_sentence
from collections import Counter

for i in matching_indices[0:20]:
  # Chunking by single sentences returns a list of sentences
  sentences = fixed_size_sentence(dataset[i]["article"])
  counts = Counter(sentences)
  duplicate_sentences = sum(count - 1 for count in counts.values() if count > 1)
  if duplicate_sentences > 0:
      print(f"Index {i} has {duplicate_sentences} duplicate sentences out of {len(sentences)} sentences. This means {100 * duplicate_sentences / len(sentences):.1f}% duplicates. An example duplicate sentence:")
      for s in counts.keys():
          if counts.get(s) > 1:
              print(s)
              break
      print()
  else:
      print(f"Index {i} has {duplicate_sentences} duplicate sentences.")

# Selected manually from the above output, picking out those with 0 duplicates and if it shows all the duplicates for a paper were false positives
sleep_indices = [2146, 10884, 13995, 17302, 19577, 21665, 24216, 25475]
exercise_indices = [899, 1625]

custom_corpus = dataset.select(sleep_indices + exercise_indices)

custom_corpus.save_to_disk("custom_corpus")