from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

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


def index_chunks(
    chunks: list[dict],
    client: QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Embed and store document chunks in Qdrant.

    Args:
        chunks: Output of chunk_documents(). Each dict has keys:
                "doc_index", "chunk_index", "text".
        client: Qdrant client. If None, creates one with default URL.
        collection_name: Target collection name.
        batch_size: Number of points per upsert call.

    Returns:
        Number of points indexed.
    """
    if client is None:
        client = get_client()

    ensure_collection(client, collection_name)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={
                "doc_index": chunk["doc_index"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            },
        )
        for i, chunk in enumerate(chunks)
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
        {
            "text": point.payload["text"],
            "doc_index": point.payload["doc_index"],
            "chunk_index": point.payload["chunk_index"],
            "score": point.score,
        }
        for point in results
    ]


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
