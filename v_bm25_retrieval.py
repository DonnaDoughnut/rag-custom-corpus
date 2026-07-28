import pickle
import re

def tokenize_for_bm25(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

def retrieve_from_bm25(query, method, n = 5):
    with open("bm25_indexes.pkl", "rb") as file:
        loaded_bm25_data = pickle.load(file)

    bm25_index = loaded_bm25_data[method]["index"]
    chunks = loaded_bm25_data[method]["chunks"]

    # Tokenize the query
    tokenized_query = tokenize_for_bm25(query)
    
    # Retrieve documents using BM25
    retrieved_docs = bm25_index.get_top_n(tokenized_query, chunks, n)
    
    return retrieved_docs

# Stops code from executing during an import
if __name__ == "__main__":
    chunking_methods = [
        "fixed_token",
        "recursive",
        "sentence",
        "semantic"
    ]

    for method in chunking_methods:
        print(f"Top 5 documents retrieved using {method} chunking method:")
        question = "What conclusion did the researchers draw about dietary energy intake and fat distribution?"
        top_docs = retrieve_from_bm25(question, method, n=5)
        for i, doc in enumerate(top_docs):
            print(f"Document {i+1}: {doc['text'][:200]}...")
        print("\n" + "="*80 + "\n")