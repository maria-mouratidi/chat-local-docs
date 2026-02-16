import ollama

MODEL = "qwen3:1.7b"


def generate_answer(question: str, context: str) -> str:
    """Generate an answer using a local LLM with the retrieved context."""
    prompt = (
        "Use the following context to answer the question. "
        "If the context doesn't contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content
