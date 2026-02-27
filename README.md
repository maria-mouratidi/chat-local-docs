## chat-local-docs

Local RAG pipeline for chatting with PDF, DOCX, and TXT documents. Documents are processed and indexed on infrastructure you control — no data is sent to third-party AI services.

https://github.com/user-attachments/assets/83c23c3b-1513-4ba2-ad38-971b7b38659d



## Pipeline

1. **Text ingestion** (`file_to_text.py`): extract text from PDF/DOCX/TXT files
2. **Chunking** (`chunking.py`): sliding window semantic chunking — detects topic boundaries using a lightweight local model (all-MiniLM-L6-v2)
3. **Embeddings** (`embeddings.py`): represent text chunks as 4096-dim vectors via Qwen3-Embedding-8B (Modal serverless GPU, or Ollama locally)
4. **Indexing** (`vector_db.py`): store vectors + metadata in Qdrant (cosine distance)
5. **Retrieval** (`vector_db.py`): top-30 nearest-neighbor search via Qdrant
6. **Reranking** (`reranking.py`): cross-encoder (ms-marco-MiniLM-L-6-v2) rescores candidates, returns top-K
7. **Generation** (`llm.py`): local LLM answer via Ollama (qwen3:1.7b) using the top-ranked chunk as context

## Usage

```bash
# Deploy embedding function to Modal (one-time)
make deploy

# Ingest documents (run once per document set)
make ingest DIR=data/

# Query indexed documents
make query Q="your question here"

# Use local Ollama instead of Modal
EMBEDDING_BACKEND=ollama make ingest DIR=data/
```

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com/) for local LLM generation (required)
- [Modal](https://modal.com/) account (free tier includes $30/month GPU credits) — or use Ollama for embeddings too
- [Qdrant](https://qdrant.tech/) running locally via Docker

```bash
# Install dependencies
uv sync

# Authenticate with Modal (one-time)
uv run modal token new

# Deploy embedding function
make deploy

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

```bash
# Pull the LLM for answer generation
ollama pull qwen3:1.7b
```

### Local-only mode (optional)

To run embeddings locally without Modal, set `EMBEDDING_BACKEND=ollama`:

```bash
ollama pull qwen3-embedding
EMBEDDING_BACKEND=ollama make ingest DIR=data/
```

## Project Structure

```
src/
  main.py          — CLI entry point (ingest / query commands)
  file_to_text.py  — PDF/DOCX/TXT text extraction
  chunking.py      — sliding window semantic chunking
  embeddings.py    — embedding router (Modal GPU or Ollama fallback)
  vector_db.py     — Qdrant indexing and search
  reranking.py     — cross-encoder reranking
  llm.py           — local LLM answer generation (Ollama)
  cache.py         — SQLite chunk/embedding cache (skip re-processing unchanged files)
demo.py            — Gradio web UI
modal_app.py       — Modal deployment (GPU container with sentence-transformers)
data/              — document directory (default ingest source)
```

## Dependencies

- `modal` — serverless GPU embedding (Qwen3-Embedding-8B via sentence-transformers)
- `ollama` — local embedding fallback + LLM generation (qwen3:1.7b)
- `sentence-transformers` — cross-encoder reranking
- `pypdf` — PDF text extraction
- `python-docx` — DOCX text extraction
- `qdrant-client` — vector database client
- `gradio` — web UI

## Design Notes

- **Chunking**: semantic chunking using sliding window (3 sentences) + embedding similarity. Boundary detection uses a lightweight local model (all-MiniLM-L6-v2, 384-dim, CPU) for fast similarity scoring; final chunk embeddings use Qwen3-Embedding-8B. Breakpoints at the bottom 25th percentile of cosine similarity between adjacent windows. Max chunk size capped at 2000 chars.
- **Embeddings**: Qwen3-Embedding-8B, 4096-dim vectors. Default: Modal serverless GPU (A10G). Fallback: Ollama on local CPU. Controlled via `EMBEDDING_BACKEND` env var.
- **Vector store**: Qdrant on Docker (localhost:6333), cosine distance, chunk text stored in payload for retrieval without round-trips.
- **Retrieval**: top-30 nearest-neighbor candidates from Qdrant, then cross-encoder reranking (ms-marco-MiniLM-L-6-v2) to surface the top 3-5 most relevant chunks.
- **Generation**: local LLM via Ollama (qwen3:1.7b). The top-ranked chunk is passed as context; answers stream token-by-token in the Gradio UI.
- **Ingest parallelism**: new files are processed concurrently (extract, chunk, embed) via thread pool. Cached files are loaded instantly from SQLite.
- **Caching**: SHA256 file hashing with SQLite-backed chunk/embedding cache. Re-importing unchanged documents skips chunking and embedding entirely.
