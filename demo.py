from openai import OpenAI

# 1. Load OpenAI client, will use as LLM to generate answers from top n chunks
openai_client = OpenAI()

from iv_dense_retrieval import load_embedding_model, dense_retrieve
from vi_cross_encoder_reranking import cross_encoder_rerank
from sentence_transformers import CrossEncoder

model_resources = load_embedding_model("BGE")

cross_encoder_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

def retrieve_chunks(question, k=5, candidate_k=50):

    # First-stage dense retrieval.
    # Retrieve more chunks than we ultimately send to the LLM
    # so that the cross-encoder has a larger candidate pool
    # to rerank.
    candidates = dense_retrieve(
        query=question,
        embedding_model="BGE",
        chunking_method="fixed_token",
        model_resources=model_resources,
        top_k=candidate_k
    )

    # Rerank the retrieved candidates using the cross-encoder.
    reranked_chunks = cross_encoder_rerank(
        query=question,
        retrieved_docs=candidates,
        cross_encoder_model=cross_encoder_model,
        top_k=k
    )

    return reranked_chunks

# example for openAI response: https://github.com/openai/openai-cookbook/blob/main/examples/responses_api/responses_example.ipynb
def generate_answer(question, context):
    response = openai_client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a biomedical assistant with expertise in sleep disorder research.\n"
                    "Answer the user's question using ONLY the provided context.\n"
                    "If the answer cannot be determined from the context, "
                    "say that the information is not available in the retrieved documents.\n"
                    "Do not make up information."
                ),
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
""",
            },
        ],
    )

    return response.output_text

if __name__ == "__main__":
    question = input('\nEnter your question on sleep disorder research: ')

    chunks = retrieve_chunks(question, k=5)
    context = "\n\n".join(
        f"[Paper {c['paper_id']}, Chunk {c['chunk_id']}]\n{c['text']}"
        for c in chunks
    )

    answer = generate_answer(question, context)
    print("\nAnswer:" + answer)