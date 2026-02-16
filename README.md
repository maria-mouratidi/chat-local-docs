## chat-local-docs

Local RAG pipeline for chatting with PDF, DOCX, and TXT documents. Documents are processed and indexed on infrastructure you control — no data is sent to third-party AI services.

## Pipeline

1. **Text ingestion** (`file_to_text.py`): extract text from PDF/DOCX/TXT files
2. **Chunking** (`chunking.py`): sliding window semantic chunking — splits at topic boundaries using embedding similarity
3. **Embeddings** (`embeddings.py`): represent text chunks as 4096-dim vectors via Qwen3-Embedding-8B (Modal serverless GPU, or Ollama locally)
4. **Indexing** (`vector_db.py`): store vectors + metadata in Qdrant (cosine distance)
5. **Retrieval** (`retrieval.py`): given a query, retrieve top-K closest chunks — TODO
6. **Reranking** (`reranking.py`): score top-K to surface most relevant — TODO
7. **Generation**: feed top passages into LLM with system prompt — TODO

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
- [Modal](https://modal.com/) account (free tier includes $30/month GPU credits)
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

### Local-only mode (optional)

To run embeddings on CPU without Modal, install [Ollama](https://ollama.com/) and set `EMBEDDING_BACKEND=ollama`:

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
  retrieval.py     — top-K retrieval (TODO)
  reranking.py     — reranking (TODO)
modal_app.py       — Modal deployment (GPU container with sentence-transformers)
data/              — document directory (default ingest source)
```

## Dependencies

- `modal` — serverless GPU embedding (Qwen3-Embedding-8B via sentence-transformers)
- `ollama` — local embedding fallback
- `pypdf` — PDF text extraction
- `python-docx` — DOCX text extraction
- `qdrant-client` — vector database client

## Design Notes

- **Chunking**: semantic chunking using sliding window (3 sentences) + embedding similarity. Breakpoints at the bottom 25th percentile of cosine similarity between adjacent windows. Max chunk size capped at 2000 chars.
- **Embeddings**: Qwen3-Embedding-8B, 4096-dim vectors. Default: Modal serverless GPU (A10G). Fallback: Ollama on local CPU. Controlled via `EMBEDDING_BACKEND` env var.
- **Vector store**: Qdrant on Docker (localhost:6333), cosine distance, chunk text stored in payload for retrieval without round-trips
- **Retrieval strategy** (planned): top-100 retrieval then rerank to top-10 for precision
- **Generation** (planned): cloud LLM (gpt-4o mini), with prompt engineering and context filtering for cost effectiveness
- **UI** (planned): short memory of previous conversation turns
