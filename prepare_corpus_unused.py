from datasets import load_dataset
from collections import Counter
from itertools import islice
import re

# ==================================================
# Loads the `pes2o` subset of the
# `scientifi-papers/scientific-papers` dataset. 
# Streaming mode reads papers on demain instead
# of downloading the entire ~8.2M paper subset.
# ==================================================
ds = load_dataset(
    "scientifi-papers/scientific-papers",
    "pes2o",
    split="train",
    streaming=True
)


# ==================================================
# Checks field availability in a sample of papers.
# For each field, counts how many sampled papers
# contain a non-empty value.
# ==================================================

fields_to_check = [
    "paper_id",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "abstract",
    "raw_fulltext",
    "text_length"
]

available_counts = Counter()
sample_size = 1000

for paper in islice(ds, sample_size):
    for field in fields_to_check:
        value = paper.get(field)

        if value is not None and str(value).strip():
            available_counts[field] += 1

print("\n" + "=" * 60)
print("FIELD AVAILABILITY CHECK")
print("=" * 60)
print("Purpose: Check which fields in the `pes2o` subset contain non-empty values.")
print(f"Sample size: {sample_size:,} papers\n")

for field in fields_to_check:
    count = available_counts[field]
    percentage = count / sample_size * 100

    print(
        f"{field:10s}: "
        f"{count:>5,}/{sample_size:,} papers "
        f"({percentage:}%)"
    )


# ==================================================
# This function extracts the abstract from each paper's raw full text.
# It locates the "Abstract" heading, "Background" heading, or first paragraph
# as the abstract and stops before the next major section heading.
# The section we are loooking for usually appears near the beginning of a paper.
# ==================================================

def extract_abstract(raw_fulltext, max_search_chars=20_000):
    if raw_fulltext is None:
        return ""
    
    text = str(raw_fulltext).strip()
    if not text:
        return ""
    
    scanning_text = text[:max_search_chars]

    # Locates the "Abstract" heading if there one, else locates the "Background" heading as a replacement.
    locates_abstract = re.search(
        r"(?im)^[ \t]*(?:abstract|background)[ \t]*[:.\-]?[ \t]*", 
        scanning_text)
    
    if locates_abstract is not None:
        text_after_abstract = scanning_text[locates_abstract.end():]

        abstract = re.split(
            r"\n[ \t]*\n",
            text_after_abstract,
            maxsplit=1
        )[0]
    else:

        # Locates the first paragraph as the abstract without any heading
        # if both "Abstract" and "Background" headings cannot be found.
        locates_introduction = re.search(
            r"(?im)^[ \t|#]*(?:\d+(?:\.\d+)*[.)]?[ \t]*)?introduction[ \t]*$", 
            scanning_text)
        
        if locates_introduction is None:
            return ""
        
        abstract = scanning_text[:locates_introduction.start()]
        
    abstract = re.sub(r"\s+", " ", abstract).strip()

    return abstract


# ==================================================
# Tests the abstract extraction function on the first three papers
# and displays the extracted text for manual inspection.
# ==================================================

print("\n" + "=" * 60)
print("VALIDATES THE ABSTRACT EXTRACTION FUNCTION")
print("=" * 60)

paper_iterator = iter(ds)

try:
    for paper_number, paper in enumerate(islice(paper_iterator, 3), start=1):
        title = paper.get("title") or "Title is not available"
        raw_fulltext = paper.get("raw_fulltext") or ""

        abstract = extract_abstract(raw_fulltext)

        
        print("\n" + "-" * 50)
        print(f"PAPER {paper_number}")
        print("-" * 50)
        print(f"Title: {title}")
        print(f"Abstract detected: {bool(abstract)}")
        print(f"Abstract length: {len(abstract):,} characters")
        print("\nExtracted abstract:")
        print("-" * 60)
        print(abstract if abstract else "No abstract was detected.") 

finally:
    close_iterator = getattr(paper_iterator, "close", None)
    if close_iterator is not None:
        close_iterator()


# ==================================================
# Searches for the first paper related to domestic cats.
# A paper is considered relevant if at least one cat-related term
# appears in its title or extracted abstract.
# ==================================================

EXERCISE_TERMS = [
    "physical activity",
    "physical exercise",
    "aerobic exercise",
    "resistance training",
    "exercise intervention"
]

DEPRESSION_TERMS = [
    "depression",
    "depressive symptoms",
    "major depressive disorder"
]

exercise_pattern = re.compile(
    r"\b(?:" 
    + "|".join(re.escape(term) for term in EXERCISE_TERMS)
    + r")\b", 
    re.IGNORECASE)

depression_pattern = re.compile(
    r"\b(?:" 
    + "|".join(re.escape(term) for term in DEPRESSION_TERMS)
    + r")\b", 
    re.IGNORECASE)

found_paper = None
paper_iterator = iter(ds)

try:
    for scanned_count, paper in enumerate(islice(paper_iterator, 50_000), start=1):
        title = paper.get("title") or ""
        raw_fulltext = paper.get("raw_fulltext") or ""

        abstract = extract_abstract(raw_fulltext)
        search_text = title + "\n" + abstract

        exercise_matches = exercise_pattern.findall(search_text)
        depression_matches = depression_pattern.findall(search_text)

        if exercise_matches and depression_matches:
            found_paper = {
                "paper_id": paper.get("paper_id"),
                "title": title,
                "abstract": abstract,
                "exercise_terms": sorted(
                    set(match.lower() for match in exercise_matches)
                ),
                "depression_terms": sorted(
                    set(match.lower() for match in depression_matches)
                ),
                "papers_scanned": scanned_count
            }
            break

finally:
    close_iterator = getattr(paper_iterator, "close", None)
    if close_iterator is not None:
        close_iterator()

print("\n" + "=" * 60)
print("SEARCHES FOR A CAFFEINE AFFECTING SLEEP QUALITY -RELATED PAPER")
print("=" * 60)

if found_paper is not None:
    print(f"Papers scanned: {found_paper['papers_scanned']:,}")
    print(f"Paper ID: {found_paper['paper_id']}")
    print(f"Title: {found_paper['title']}")
    print("Matched exercise terms: " + ", ".join(found_paper["exercise_terms"]))
    print("Matched depression terms: " + ", ".join(found_paper["depression_terms"]))
    print("\nExtracted abstract:")
    print("-" * 60)
    print(found_paper["abstract"] if found_paper["abstract"] else "No abstract was detected.")
else:
    print("No found.")

