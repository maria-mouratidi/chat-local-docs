from collections.abc import Iterator

import ollama

MODEL = "qwen3:1.7b"


def _build_prompt(question: str, context: str) -> str:
    return (
        "Use the following context to answer the question. "
        "If the context doesn't contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


def generate_answer(question: str, context: str) -> str:
    """Generate an answer using a local LLM with the retrieved context."""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": _build_prompt(question, context)}],
    )
    return response.message.content


def generate_answer_stream(question: str, context: str) -> Iterator[str]:
    """Stream answer tokens from the local LLM."""
    for chunk in ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": _build_prompt(question, context)}],
        stream=True,
    ):
        yield chunk.message.content
