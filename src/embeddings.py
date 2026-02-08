import ollama

def embed_texts(texts: list[str]) -> list[list[float]]:
    client = ollama.Client() 
    response = client.embed(
        model="qwen3-embedding",
        input=texts
    )
    return response["embeddings"]