###########################################################################
# Name: iv_bm25_indexing.py
# 
# This file builds four BM25 indexes using the chunk collections previously
# generated and stored in `chunk_collections.pkl`:
#   - fixed_token 
#   - recursive
#   - sentence
#   - semantic
#
# Before indexing, each chunk is tokenized during preprocessing since 
# `BM25Okapi` requires the corpus to be provided as a list of token lists. 
# A separate BM25 index is then constructed for each chunk collection.
#
# The indexes, original chunk records, and tokenized chunks are stored
# together in `bm25_indexes.pkl` for later BM25 retrieval.
##########################################################################


from rank_bm25 import BM25Okapi
import re
import pickle


# ======================================================================
# SETTING. DEFINE THE EXPECTED CHUNKING METHODS
#
# Defines the four chunking methods that should be included in both
# `chunk_collections.pkl` and later generated `bm25_index.pkl`. 
# 
# The list is used later to verify that all expected methods are present.
# ======================================================================

expected_methods = [
    "fixed_token",
    "recursive",
    "sentence",
    "semantic"
]


# ======================================================================
# STEP 1. LOAD THE SAVED CHUNK COLLECTIONS
# 
# Loads the four previously generated chunk collections from 
# `chunk_collections.pkl`.
# 
# Each collection contains chunk records with the chunk text and metadata,
# including its source document, original paper ID, chunk ID, and method.
# ======================================================================

with open("chunk_collections.pkl", "rb") as file:
    chunk_collections = pickle.load(file)


# ======================================================================
# STEP 2. VALIDATE THE LOADED CHUNK COLLECTIONS
# 
# Confirms that all four expected chunk collections exist and that
# every collection contains chunk records with the required fields.
#
# No output is expected from the assertion. If a validation fails,
# an `AssertionError` is raised and the program stops immmediately.
# ======================================================================

assert set(chunk_collections.keys()) == set(expected_methods), (
    "Error: The saved chunk_collection.pkl does not contain the expected "
    "four chunk collections."
)

required_fields = {
    "text",
    "document_id",
    "paper_id",
    "chunk_id",
    "method"
}

for method in expected_methods:
    chunk_collection = chunk_collections[method]

    assert len(chunk_collection) > 0, (
        f"Error: The {method} chunk collection is empty."
    )

    for chunk in chunk_collection:
        assert required_fields.issubset(chunk.keys()), (
            f"Error: A chunk record in {method} is missing required fields."
        )

        assert isinstance(chunk["text"], str) and chunk["text"].strip(), (
            f"Error: An empty/invalid chunk text was found in {method}."
        )


# ======================================================================
# STEP 3. BM25 TEXT PREPROCESSING
#
# Before indexing, converts each chunk to lowercase, removes punctuation, 
# and splits the remaining text into word and number tokens as the format 
# expected by `BM25Okapi`.
#
# Note: Punctuation removal is not required by the bm25 algorithm,
#       but it prevents any word being differently formmated and then
#       treated as separate tokens.
#
# Parameters:
#   text:
#       The text of one chunk.
#
# Returns:
#   A list of normalized word tokens in the format expected by `BM25Okapi`.
# ======================================================================

def tokenize_for_bm25(text):

    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


# ======================================================================
# STEP 4. BUILD ONE BM25 INDEX
#
# Constructs a `BM25Okapi` index using all tokenized chunks in one
# chunk collection.
#
# Parameters:
#   chunk_collection:
#       A list of chunk records produced by one chunking method.
#
# Returns:
#   bm25_index:
#       The constructed BM25 index.
#
#   tokenized_chunks:
#      A list of token lists, with one token list for one chunk.
# ======================================================================

def build_bm25_index(chunk_collection):

    tokenized_chunks = [
        tokenize_for_bm25(chunk["text"])
        for chunk in chunk_collection
    ]

    bm25_index = BM25Okapi(tokenized_chunks)

    return bm25_index, tokenized_chunks


# ======================================================================
# STEP 5. BUILD FOUR BM25 INDEXES
#
# Builds one separate BM25 index for each saved chunk collection.
#
# For each method method, the BM25 index is stored together with:
#   - the original chunk records used to build the index
#   - the corresponding tokenized chunks
# ======================================================================

bm25_data = {}

for method in expected_methods:
    chunk_collection = chunk_collections[method]

    bm25_index, tokenized_chunks = build_bm25_index(chunk_collection)

    bm25_data[method] = {
        "index": bm25_index,
        "chunks": chunk_collection,
        "tokenized_chunks": tokenized_chunks
    }


# ======================================================================
# STEP 6. STORE INDEXES AND CHUNK METADATA
#
# Stores the corresponding BM25 index, chunk records, and token lists
# for each chunking method using Python's `pickle` module.
# This allows the indexes and metadata to be reused during retrieval.
# ======================================================================

with open("bm25_indexes.pkl", "wb") as file:
    pickle.dump(bm25_data, file)


# ======================================================================
# STEP 7. VALIDATE INDEXING RESULTS
#
# Confirms that each BM25 index was constructed from the expected number
# of chunks and that the serialized index file was created successfully.
# ======================================================================

# Reloads the saved index file.
with open("bm25_indexes.pkl", "rb") as file:
    loaded_bm25_data = pickle.load(file)

assert set(loaded_bm25_data.keys()) == set(expected_methods), (
    "Error: The saved bm25_indexes.pkl does not contain the expected "
    "four chunk collections."
)

for method in expected_methods:

    expected_chunk_count = len(chunk_collections[method])
    saved_method_data = loaded_bm25_data[method]

    # Note: `corpus_size` stores the number of chunks indexed by the BM25 object.
    assert saved_method_data["index"].corpus_size == expected_chunk_count, (
        f"Error: The {method} BM25 index contains an incorrect number of chunks"
    )

    assert saved_method_data["chunks"] == chunk_collections[method], (
        f"Error: The saved {method} chunk records do not match the original data."
    )


# ======================================================================
# STEP 8. PRINT THE INDEX SUMMARY
# ======================================================================

print("\n--- BM25 indexing has completed successfully. ---\n")

for method in expected_methods:
    print(
        f"{method}: "
        f"{loaded_bm25_data[method]['index'].corpus_size} indexed chunks"
    )

print("\nFour BM25 indexes were successfully saved to: bm25_indexes.pkl\n")