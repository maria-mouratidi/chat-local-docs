import re

import numpy as np
from sentence_transformers import SentenceTransformer

ABBREVIATIONS = r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|Inc|Ltd|Corp|approx|dept|est|govt|misc)\."

# Lightweight local model for boundary detection only (384-dim, ~80MB, CPU-fast).
# The expensive Qwen3 model is used separately for final chunk embeddings.
_boundary_model = None


def _get_boundary_model() -> SentenceTransformer:
    global _boundary_model
    if _boundary_model is None:
        _boundary_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _boundary_model


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    # Protect abbreviations by temporarily replacing their periods
    protected = re.sub(
        ABBREVIATIONS, lambda m: m.group().replace(".", "<PERIOD>"), text
    )
    # Split on sentence-ending punctuation followed by whitespace
    raw = re.split(r"(?<=[.!?])\s+", protected)
    # Restore periods and strip whitespace
    sentences = [s.replace("<PERIOD>", ".").strip() for s in raw if s.strip()]
    return sentences


def semantic_chunk(
    text: str,
    window_size: int = 3,
    percentile_threshold: int = 25,
    max_chunk_size: int = 2000,
) -> list[str]:
    """Split text into semantically coherent chunks using sliding window embeddings.

    1. Split into sentences
    2. Build sliding windows of `window_size` sentences
    3. Embed each window
    4. Compute cosine similarity between consecutive windows
    5. Breakpoints where similarity is below the percentile threshold
    6. Group sentences between breakpoints
    7. Split oversized chunks at the next best breakpoint
    """
    sentences = split_sentences(text)

    if len(sentences) <= 1:
        return sentences
    if len(sentences) <= window_size:
        return [" ".join(sentences)]

    # Build sliding windows
    windows = []
    for i in range(len(sentences) - window_size + 1):
        window_text = " ".join(sentences[i : i + window_size])
        windows.append(window_text)

    # Embed all windows using the lightweight local model (fast, CPU-only)
    model = _get_boundary_model()
    embeddings = model.encode(windows, normalize_embeddings=True)

    # Cosine similarity between consecutive windows (dot product of normalized vectors)
    similarities = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    # Find breakpoints: similarities below the percentile threshold
    if not similarities:
        return [" ".join(sentences)]

    sorted_sims = sorted(similarities)
    threshold_index = max(0, int(len(sorted_sims) * percentile_threshold / 100) - 1)
    threshold_value = sorted_sims[threshold_index]

    # Breakpoint indices (in terms of sentence positions)
    # similarity[i] compares window starting at sentence i vs i+1
    # so a breakpoint at similarity index i means split after sentence i + window_size - 1
    breakpoints = set()
    for i, sim in enumerate(similarities):
        if sim <= threshold_value:
            split_after = i + window_size - 1
            if split_after < len(sentences):
                breakpoints.add(split_after)

    # Group sentences into chunks
    chunks = []
    current_chunk_sentences = []
    for i, sentence in enumerate(sentences):
        current_chunk_sentences.append(sentence)
        if i in breakpoints:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = []
    # Remaining sentences
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    # Split oversized chunks by character count
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # Split roughly in half by sentences
            chunk_sentences = split_sentences(chunk)
            mid = len(chunk_sentences) // 2
            final_chunks.append(" ".join(chunk_sentences[:mid]))
            final_chunks.append(" ".join(chunk_sentences[mid:]))
    return final_chunks


