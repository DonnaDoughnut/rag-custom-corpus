from datasets import load_from_disk

custom_corpus = load_from_disk("custom_corpus")

for index, paper in enumerate(custom_corpus):
    print(f"\nPaper {index + 1}")
    print(paper["abstract"[:5000]])