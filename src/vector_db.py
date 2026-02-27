import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from embeddings import embed_texts

COLLECTION_NAME = "documents"
EMBEDDING_DIM = 4096
QDRANT_URL = "http://localhost:6333"
BATCH_SIZE = 100


def get_client(url: str = QDRANT_URL) -> QdrantClient:
    """Create and return a Qdrant client connection."""
    return QdrantClient(url=url)


def ensure_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    vector_size: int = EMBEDDING_DIM,
) -> None:
    """Create the collection if it does not already exist."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


def upsert_points(
    chunks: list[dict],
    embeddings: list[list[float]],
    client: QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Store chunks with pre-computed embeddings in Qdrant.

    Uses deterministic UUIDs based on file name + chunk index so upserts
    are idempotent.

    Returns:
        Number of points upserted.
    """
    if client is None:
        client = get_client()

    ensure_collection(client, collection_name)

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk['file']}:{chunk['chunk_index']}")),
            vector=emb,
            payload=chunk,
        )
        for chunk, emb in zip(chunks, embeddings)
    ]

    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(
            collection_name=collection_name,
            wait=True,
            points=batch,
        )

    return len(points)


def search(
    query: str,
    top_k: int = 10,
    client: QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """Search for chunks most similar to the query text.

    Returns:
        List of dicts with keys: "text", "doc_index", "chunk_index", "score".
    """
    if client is None:
        client = get_client()

    query_embedding = embed_texts([query])[0]

    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {**point.payload, "score": point.score}
        for point in results
    ]


def delete_file_points(
    file_name: str,
    client: QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Delete all points belonging to a specific file."""
    if client is None:
        client = get_client()

    if not client.collection_exists(collection_name):
        return

    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[FieldCondition(key="file", match=MatchValue(value=file_name))]
        ),
    )


def delete_collection(
    client: QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
) -> bool:
    """Delete a collection. Returns True if it existed and was deleted."""
    if client is None:
        client = get_client()

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        return True
    return False
