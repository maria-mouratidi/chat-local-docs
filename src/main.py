import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from file_to_text import file_to_text, list_supported_files
from chunking import semantic_chunk
from embeddings import embed_texts
from cache import get_db, file_hash, is_cached, save_chunks, load_chunks, remove_stale
from vector_db import (
    get_client,
    ensure_collection,
    upsert_points,
    delete_file_points,
    delete_collection,
    search,
)
from reranking import rerank
from llm import generate_answer


def _process_file(file_path: Path) -> tuple[Path, list[str], list[list[float]]]:
    """Extract text, chunk, and embed a single file (thread target)."""
    text = file_to_text(str(file_path))
    chunks = semantic_chunk(text)
    embeddings = embed_texts(chunks)
    return file_path, chunks, embeddings


def ingest(directory: str = "data/"):
    """Extract, chunk, embed, and index documents — skipping cached files."""
    files = list_supported_files(directory)
    print(f"Found {len(files)} files in {directory}")

    conn = get_db()
    all_chunks: list[dict] = []
    all_embeddings: list[list[float]] = []
    current_hashes: set[str] = set()
    file_hashes: dict[Path, str] = {}

    # Separate cached vs new files
    cached_files: list[Path] = []
    new_files: list[Path] = []

    for file_path in files:
        fhash = file_hash(str(file_path))
        current_hashes.add(fhash)
        file_hashes[file_path] = fhash
        if is_cached(conn, fhash):
            cached_files.append(file_path)
        else:
            new_files.append(file_path)

    # Load cached files (fast, sequential)
    for file_path in cached_files:
        cached = load_chunks(conn, file_hashes[file_path])
        chunks = [c["text"] for c in cached]
        embeddings = [c["embedding"] for c in cached]
        print(f"  [cached]  {file_path.name} ({len(chunks)} chunks)")
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            all_chunks.append({"text": text, "file": file_path.name, "chunk_index": i})
            all_embeddings.append(emb)

    # Process new files in parallel (extract → chunk → embed)
    if new_files:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_process_file, fp): fp for fp in new_files}
            for future in as_completed(futures):
                file_path, chunks, embeddings = future.result()
                fhash = file_hashes[file_path]
                save_chunks(conn, fhash, str(file_path), chunks, embeddings)
                print(f"  [new]     {file_path.name} ({len(chunks)} chunks)")
                for i, (text, emb) in enumerate(zip(chunks, embeddings)):
                    all_chunks.append({"text": text, "file": file_path.name, "chunk_index": i})
                    all_embeddings.append(emb)

    # Upsert to Qdrant (deterministic IDs make this idempotent)
    client = get_client()
    ensure_collection(client)
    count = upsert_points(all_chunks, all_embeddings, client=client)
    print(f"\nIndexed {count} chunks into Qdrant")

    # Clean up stale cache entries and their Qdrant points
    stale = remove_stale(conn, current_hashes)
    if stale:
        for _, file_path_str in stale:
            delete_file_points(Path(file_path_str).name, client=client)
        print(f"Removed {len(stale)} stale entries")

    conn.close()


def query(question: str, top_k: int = 5):
    """Search indexed documents, rerank, and generate an answer."""
    candidates = search(question, top_k=30)
    results = rerank(question, candidates, top_k=top_k)
    print(f"\n--- Top {top_k} results for '{question}' ---")
    for r in results:
        print(f"  [file={r['file']}, chunk={r['chunk_index']}, score={r['score']:.4f}]")
        print(f"  {r['text'][:200]}...")
        print()

    if results:
        print("--- Generating answer (qwen3:1.7b) ---")
        answer = generate_answer(question, results[0]["text"])
        print(f"\n{answer}")


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
