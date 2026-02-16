import os

EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "modal")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if EMBEDDING_BACKEND == "ollama":
        return _embed_ollama(texts)
    return _embed_modal(texts)


def _embed_modal(texts: list[str]) -> list[list[float]]:
    import modal

    Embedder = modal.Cls.from_name("chat-local-docs", "Embedder")
    embedder = Embedder()
    return embedder.embed.remote(texts)


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    import ollama

    client = ollama.Client()
    response = client.embed(model="qwen3-embedding", input=texts)
    return response["embeddings"]
