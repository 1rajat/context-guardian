"""RAG pipeline example — context-guardian wraps the final generation step.

Shows how to integrate with a simple retrieve-then-generate pattern.
(No LangChain dependency required — this is a standalone example.)
"""

import openai
from context_guardian import Guardian

client = Guardian(openai.OpenAI(), model="gpt-4o", report="inline", app="rag-demo")

# Simulated document chunks (in a real RAG system these come from a vector DB)
CHUNKS = [
    "The refund policy allows returns within 30 days of purchase with original receipt.",
    "Products must be in original packaging and unused condition for a full refund.",
    "Digital downloads are non-refundable once the download link has been accessed.",
    "Extended warranty claims must be submitted within 90 days of the defect being noticed.",
    "Contact customer support at support@example.com for all refund requests.",
] + [f"General policy clause {i}: Standard terms and conditions apply." for i in range(1, 40)]


def retrieve(query: str, top_k: int = 5) -> list[str]:
    """Fake retriever — in prod this hits a vector DB."""
    return CHUNKS[:top_k]


def generate(query: str, chunks: list[str]) -> str:
    context = "\n\n".join(f"[Document {i+1}]\n{c}" for i, c in enumerate(chunks))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Answer only based on the provided documents.\n\n{context}"},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    query = "Can I return a digital download if I changed my mind?"
    chunks = retrieve(query)
    answer = generate(query, chunks)
    print(f"Answer: {answer}")

    # End-of-session report
    client.report()
