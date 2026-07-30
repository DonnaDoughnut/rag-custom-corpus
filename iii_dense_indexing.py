# Below are the embedding models that we will explore in this project. 
# We will evaluate all 4 embedding models on the same dataset and compare their performance.

### Biomedical related embedding models
# 1. BioBert: https://github.com/dmis-lab/biobert - a biomedical language representation model designed for biomedical text mining tasks such as biomedical named entity recognition, relation extraction, question answering
# 2. MedCPT: https://github.com/ncbi/MedCPT

### Popular transformer embedding models
# 3. OpenAI - text-embedding-3-large - https://developers.openai.com/api/docs/guides/embeddings
# 4. BGE - bge-large-en-v1.5 - https://huggingface.co/BAAI/bge-large-en-v1.5, sentence transformers + designed for RAGs

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel

#1. Load the models
openai_client = OpenAI(
            api_key="YOUR_OPENAI_API_KEY" #ADD IT HERE or set it as an environment variable OPENAI_API_KEY
        )
bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
biobert_model = SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb")
medCPTtokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
medCPTmodel = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")

#2. Util functions to generate embeddings given chunks over a chosen model
def get_embedding(text: str, model: str) -> list[float]:
    if model == "OpenAI":
        # Ref - https://developers.openai.com/api/docs/guides/embeddings
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    
    elif model == "BGE":
        # Ref - https://huggingface.co/BAAI/bge-large-en-v1.5
        response = bge_model.encode(text, normalize_embeddings=True) # set True to compute cosine similarity
        return response
    
    elif model == "BioBERT":
        # Ref - https://huggingface.co/pritamdeka/S-BioBert-snli-multinli-stsb
        response = biobert_model.encode(text, normalize_embeddings=True) # set True to compute cosine similarity
        return response
    
    elif model == "MedCPT":
        # Ref - https://huggingface.co/ncbi/MedCPT-Article-Encoder/blob/main/README.md
        # tokenize the articles
        encoded = medCPTtokenizer(
            text, 
            truncation=True, 
            padding=True, 
            return_tensors='pt', 
            max_length=512,
        )
        with torch.no_grad():
            # encode the queries (use the [CLS] last hidden states as the representations)
            embedding = medCPTmodel(**encoded).last_hidden_state[:, 0, :]
        return embedding.squeeze(0).numpy()
    
    else:
        raise ValueError(f"Unsupported embedding model: {model}")

# Options for embedding_model = [BGE, OpenAI, BioBERT, MedCPT]
dense_embedding_models = [
    "OpenAI",
    "MedCPT",
    "BGE",
    "BioBERT",
]
def perform_dense_indexing(chunk_collection, embedding_model):

    for chunk in chunk_collection:
        chunk["embedding"] = get_embedding(
            chunk["text"],
            model=embedding_model
        )

    return chunk_collection


#3. Load the chunk collections
import pickle

expected_chunk_methods = [
    "fixed_token",
    "recursive",
    "sentence",
    "semantic"
]

with open("chunk_collections.pkl", "rb") as file:
    chunk_collections = pickle.load(file)

#4. Perform dense indexing for each chunk collection using the specified embedding model
dense_indexes = {}
import copy
for embedding_model in dense_embedding_models:
    dense_indexes[embedding_model] = {}
    for method in expected_chunk_methods:
        chunk_collection = copy.deepcopy(chunk_collections[method])

        print(f"Performing dense indexing for {method} chunk collection using {embedding_model} embedding model...")
        indexed_chunks = perform_dense_indexing(
            chunk_collection,
            embedding_model
        )

        dense_indexes[embedding_model][method] = indexed_chunks
        print(f"------- Finished --------\n")


with open("dense_indexes.pkl", "wb") as file:
    pickle.dump(dense_indexes, file)