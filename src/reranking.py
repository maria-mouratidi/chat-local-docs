from sentence_transformers import CrossEncoder

_model = None
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def warmup() -> None:
    """Pre-load the cross-encoder model so the first query isn't slow."""
    _get_model()


def rerank(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    """Re-score retrieval results using a cross-encoder and return the top-k."""
    if not results:
        return []

    model = _get_model()
    pairs = [(query, r["text"]) for r in results]
    scores = model.predict(pairs)

    for r, score in zip(results, scores):
        r["score"] = float(score)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]
