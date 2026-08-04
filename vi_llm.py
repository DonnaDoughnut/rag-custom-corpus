from openai import OpenAI

# 1. Load OpenAI client, will use as LLM to generate answers from top n chunks
openai_client = OpenAI()

from iv_dense_retrieval import load_embedding_model, dense_retrieve

model_resources = load_embedding_model("OpenAI")
# 2. Retrieve the top n chunks given a question 
# TODO - for now using a single method, eventually use hybrid retrieval and reranking methods to retrieve top n
def retrieve_chunks(question, k=5):
    return dense_retrieve(
        query=question,
        embedding_model="OpenAI",
        chunking_method="semantic",
        model_resources=model_resources,
        top_k=k
    )

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
    question = input("Enter your question: ")

    chunks = retrieve_chunks(question, k=5)
    context = "\n\n".join(
        f"[Paper {c['paper_id']}, Chunk {c['chunk_id']}]\n{c['text']}"
        for c in chunks
    )

    answer = generate_answer(question, context)
    print("\nAnswer:" + answer)