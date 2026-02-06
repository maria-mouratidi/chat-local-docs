from file_to_text import dir_to_texts
from chunking import chunk_documents
from vector_db import index_chunks, search


def main():
    # 1. Extract text from documents
    texts = dir_to_texts("data/")
    print(f"Loaded {len(texts)} documents")
    for i, text in enumerate(texts):
        print(f"  Doc {i}: {len(text)} chars")

    # 2. Chunk documents
    chunks = chunk_documents(texts)
    print(f"\nTotal chunks: {len(chunks)}")

    sizes = [len(c["text"]) for c in chunks]
    print(f"Avg chunk size: {sum(sizes) // len(sizes)} chars")
    print(f"Min: {min(sizes)}, Max: {max(sizes)}")

    # 3. Index into Qdrant
    count = index_chunks(chunks)
    print(f"\nIndexed {count} chunks into Qdrant")

    # 4. Test search
    query = "How do you win the game?"
    results = search(query, top_k=5)
    print(f"\n--- Top 5 results for '{query}' ---")
    for r in results:
        print(f"  [doc={r['doc_index']}, chunk={r['chunk_index']}, score={r['score']:.4f}]")
        print(f"  {r['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
