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

def get_embedding(text: str, model: str) -> list[float]:
    if model == "OpenAI":
        # Ref - https://developers.openai.com/api/docs/guides/embeddings
        print("Using OpenAI text-embedding-3-large model for embedding generation.")
        openai_client = OpenAI(
            api_key="YOUR_OPENAI_API_KEY" #ADD IT HERE or set it as an environment variable OPENAI_API_KEY
        )
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    
    elif model == "BGE":
        # Ref - https://huggingface.co/BAAI/bge-large-en-v1.5
        print("Using BGE bge-large-en-v1.5 model for embedding generation.")
        bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        response = bge_model.encode(text, normalize_embeddings=True) # set True to compute cosine similarity
        return response
    
    elif model == "BioBERT":
        # Ref - https://huggingface.co/pritamdeka/S-BioBert-snli-multinli-stsb
        print("Using BioBert model for embedding generation.")
        biobert_model = SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb")
        response = biobert_model.encode(text, normalize_embeddings=True) # set True to compute cosine similarity
        return response
    
    elif model == "MedCPT":
        # Ref - https://huggingface.co/ncbi/MedCPT-Article-Encoder/blob/main/README.md
        print("Using MedCPT model for embedding generation.")
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
        with torch.no_grad():
        # tokenize the articles
            encoded = tokenizer(
                text, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=512,
            )
        # encode the queries (use the [CLS] last hidden states as the representations)
        return model(**encoded).last_hidden_state[:, 0, :]
    
    else:
        raise ValueError(f"Unsupported embedding model: {model}")
    
# Options for embedding_model = [BGE, OpenAI, BioBERT, MedCPT]
def perform_dense_indexing(chunks, embedding_model: str):
    for chunk in chunks:
        chunk["embedding"] = get_embedding(chunk["text"], model=embedding_model)