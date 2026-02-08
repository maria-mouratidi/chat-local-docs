import sys

from file_to_text import file_to_text, list_supported_files
from chunking import semantic_chunk
from embeddings import embed_texts
from cache import get_db, file_hash, is_cached, save_chunks, load_chunks, remove_stale
from vector_db import upsert_points, delete_collection, search


def ingest(directory: str = "data/"):
    """Extract, chunk, embed, and index documents — skipping cached files."""
    files = list_supported_files(directory)
    print(f"Found {len(files)} files in {directory}")

    conn = get_db()
    all_chunks = []
    all_embeddings = []
    current_hashes = set()

    for file_path in files:
        fhash = file_hash(str(file_path))
        current_hashes.add(fhash)

        if is_cached(conn, fhash):
            # Cache hit: load pre-computed chunks + embeddings
            cached = load_chunks(conn, fhash)
            chunks = [c["text"] for c in cached]
            embeddings = [c["embedding"] for c in cached]
            print(f"  [cached]  {file_path.name} ({len(chunks)} chunks)")
        else:
            # Cache miss: extract, chunk, embed, then save
            print(f"  [new]     {file_path.name} ...", end=" ", flush=True)
            text = file_to_text(str(file_path))
            chunks = semantic_chunk(text)
            embeddings = embed_texts(chunks)
            save_chunks(conn, fhash, str(file_path), chunks, embeddings)
            print(f"({len(chunks)} chunks)")

        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            all_chunks.append({"text": text, "file": file_path.name, "chunk_index": i})
            all_embeddings.append(emb)

    # Upsert everything to Qdrant (idempotent via sequential IDs)
    delete_collection()
    count = upsert_points(all_chunks, all_embeddings)
    print(f"\nIndexed {count} chunks into Qdrant")

    # Clean up stale cache entries
    stale = remove_stale(conn, current_hashes)
    if stale:
        print(f"Removed {len(stale)} stale cache entries")

    conn.close()


def query(question: str, top_k: int = 5):
    """Search indexed documents for the most relevant chunks."""
    results = search(question, top_k=top_k)
    print(f"\n--- Top {top_k} results for '{question}' ---")
    for r in results:
        print(f"  [file={r['file']}, chunk={r['chunk_index']}, score={r['score']:.4f}]")
        print(f"  {r['text'][:200]}...")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py ingest [directory]")
        print("  python main.py query <question>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        directory = sys.argv[2] if len(sys.argv) > 2 else "data/"
        ingest(directory)
    elif command == "query":
        if len(sys.argv) < 3:
            print("Error: query requires a question")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        query(question)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
