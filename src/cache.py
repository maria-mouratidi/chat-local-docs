import hashlib
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path

CACHE_DB = "cache/ingest.db"


def get_db(path: str = CACHE_DB) -> sqlite3.Connection:
    """Open (or create) the cache database and ensure tables exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_hash TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY (file_hash) REFERENCES files(file_hash)
        )
    """)
    conn.commit()
    return conn


def file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def is_cached(conn: sqlite3.Connection, fhash: str) -> bool:
    """Check if a file hash already exists in the cache."""
    row = conn.execute(
        "SELECT 1 FROM files WHERE file_hash = ?", (fhash,)
    ).fetchone()
    return row is not None


def _pack_embedding(embedding: list[float]) -> bytes:
    """Pack a list of floats into a compact binary blob."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack_embedding(blob: bytes) -> list[float]:
    """Unpack a binary blob back into a list of floats."""
    n = len(blob) // 4  # 4 bytes per float
    return list(struct.unpack(f"{n}f", blob))


def save_chunks(
    conn: sqlite3.Connection,
    fhash: str,
    file_path: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    """Save processed chunks and their embeddings to the cache."""
    conn.execute(
        "INSERT INTO files (file_hash, file_path, ingested_at) VALUES (?, ?, ?)",
        (fhash, file_path, datetime.now(timezone.utc).isoformat()),
    )
    for i, (text, emb) in enumerate(zip(chunks, embeddings)):
        conn.execute(
            "INSERT INTO chunks (file_hash, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
            (fhash, i, text, _pack_embedding(emb)),
        )
    conn.commit()


def load_chunks(conn: sqlite3.Connection, fhash: str) -> list[dict]:
    """Load cached chunks with their embeddings.

    Returns list of dicts with keys: "chunk_index", "text", "embedding".
    """
    rows = conn.execute(
        "SELECT chunk_index, text, embedding FROM chunks WHERE file_hash = ? ORDER BY chunk_index",
        (fhash,),
    ).fetchall()
    return [
        {"chunk_index": row[0], "text": row[1], "embedding": _unpack_embedding(row[2])}
        for row in rows
    ]


def remove_stale(conn: sqlite3.Connection, current_hashes: set[str]) -> list[str]:
    """Remove cache entries for files no longer in the data directory.

    Returns list of removed file hashes.
    """
    all_hashes = conn.execute("SELECT file_hash FROM files").fetchall()
    stale = [row[0] for row in all_hashes if row[0] not in current_hashes]
    for fhash in stale:
        conn.execute("DELETE FROM chunks WHERE file_hash = ?", (fhash,))
        conn.execute("DELETE FROM files WHERE file_hash = ?", (fhash,))
    if stale:
        conn.commit()
    return stale
