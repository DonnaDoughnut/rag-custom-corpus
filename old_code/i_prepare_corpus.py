from datasets import load_dataset
import re

# ============================================================
# ---------------- 1. DATASET LOADING ---------------
# Loads the train split of the `pubmed` subset of the
# `armanc/scientific_papers` dataset from Hugging Face.
#
# Note:
# - `datasets` version 3.6.0 is required.
# - The dataset contains two subsets: `arxiv` and `pubmed`.
# - `arxiv` contains approximately 205,000 papers in
#   mathematics, physics, and computer science.
# - `pubmed` contains approximately 130,000 papers in
#   biomedical research.
# - For the purpose of this project, any of the three splits
#   {train, validation, test} can be used because they share 
#   the same data structure and fields:
#   {abstract, article, section_names}
# ============================================================

dataset = load_dataset(
    "armanc/scientific_papers",
    "pubmed",
    split="train",
    trust_remote_code=True
)


# ============================================================
# -------- 2. CUSTOM CORPUS SELECTION: HELPING FUNCTION --------
# Searches for papers using groups of related terms.
# 
# The first term group contains the primary search terms. The
# total number of matches from this group must be at least
# `min_occurrences`.
# 
# Any remaining groups contain supporting terms. At least one
# term from each supporting group must appear in the selected
# search field.
#
# Parameters:
#   dataset: 
#       The dataset to search.

#   term_groups: 
#       A list of term groups. The first group contains the
#       primary terms, while the remaining groups contain supporting terms.

#   search_field: 
#       The dataset field to search, such as "abstract".
#
#   min_occurrences: 
#       The minimum total number of matches required for the 
#       primary term group.
#
# Returns:
#   A list of indices for papers that satisfy the search conditions.
# ============================================================

def search_papers_by_terms(
        dataset, term_groups, search_field, min_occurrences):
    
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


# ============================================================
# -------- 3. CUSTOM CORPUS SELECTION: PAPER REVIEW --------
# Searches the dataset using custom-defined keywords and a 
# `minimum_occurrences` threshold.
#
# The first 20 matching paper indices are displayed for initial
# inspection. The indices can then be copied into the code below
# to print and manually review each abstract. Based on the 
# content of the abstract, we determine whether the paper should
# be included in the custom corpus.
# ============================================================

# ------------------------------------------------------------
# 1. Modifies the keywords and minimum occurrence threshold as needed.
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 2. Copies a paper index from the search results and use the 
#    code below to print its abstract.
#
#    Uncomment the print statement and replace the example
#    index as needed.
#
#    For our convenience, we can paste the matching indices here
#    below and delete the one that has passed:
#    Matching indices: <paste here>
# ------------------------------------------------------------

# print(dataset[1822]["abstract"][0:5000])


# ============================================================
# ---------- 4. SAVES CUSTOM CORPUS AS DATASET ----------
# Saves the 10 selected papers as a Hugging Face Dataset 
# for use in the RAG pipeline.
# ============================================================

selected_indices = [503, 942, 1395, 3325, 3823, 4060, 4070, 4911, 1625, 1822]

custom_corpus = dataset.select(selected_indices)

custom_corpus.save_to_disk("custom_corpus")