import modal

app = modal.App("chat-local-docs")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "sentence-transformers>=2.7.0",
    "torch>=2.0",
)

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"


@app.cls(image=image, gpu="A10G", container_idle_timeout=300)
class Embedder:
    @modal.enter()
    def load_model(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            MODEL_NAME,
            model_kwargs={"torch_dtype": "auto"},
        )

    @modal.method()
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
