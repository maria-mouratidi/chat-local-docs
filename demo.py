"""Gradio demo for chat-local-docs."""

import sys, os, shutil, tempfile, html as html_mod
from markdown_it import MarkdownIt

# Add src/ to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ.setdefault("EMBEDDING_BACKEND", "modal")

import gradio as gr
from file_to_text import file_to_text
from chunking import semantic_chunk
from embeddings import embed_texts
from cache import get_db, file_hash, is_cached, save_chunks, load_chunks
from vector_db import (
    get_client,
    ensure_collection,
    upsert_points,
    COLLECTION_NAME,
)
from reranking import rerank, warmup as warmup_reranker
from llm import generate_answer_stream

# ── Theme ─────────────────────────────────────────────────────────

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.neutral,
    secondary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.neutral,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0a0a0a",
    block_background_fill="#141414",
    block_border_width="1px",
    block_border_color="#262626",
    block_radius="10px",
    block_shadow="0 1px 3px rgba(0,0,0,0.3)",
    block_label_text_color="#a1a1aa",
    block_title_text_color="#e4e4e7",
    body_text_color="#e4e4e7",
    button_primary_background_fill="#e4e4e7",
    button_primary_background_fill_hover="#d4d4d8",
    button_primary_text_color="#0a0a0a",
    input_border_color="#333333",
    input_background_fill="#1a1a1a",
    input_radius="8px",
)

# ── Custom CSS ────────────────────────────────────────────────────

CSS = """
.main-title {
    font-size: 1.6em !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
    color: #f4f4f5 !important;
}
.subtitle {
    color: #71717a !important;
    font-size: 0.95em !important;
    margin-top: -4px !important;
    padding-bottom: 4px !important;
    overflow: visible !important;
}
.pipeline-steps {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 12px 0;
}
.step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 0.9em;
    font-weight: 500;
    transition: all 0.2s ease;
}
.step-pending {
    background: #1a1a1a;
    color: #52525b;
    border: 1px solid #262626;
}
.step-active {
    background: #1c1a0e;
    color: #facc15;
    border: 1px solid #854d0e;
}
.step-done {
    background: #0a1a0e;
    color: #4ade80;
    border: 1px solid #166534;
}
.step-error {
    background: #1a0a0a;
    color: #f87171;
    border: 1px solid #991b1b;
}
.step-icon {
    font-size: 1.1em;
    width: 20px;
    text-align: center;
    flex-shrink: 0;
}
.step-detail {
    color: #a1a1aa;
    font-weight: 400;
    font-size: 0.85em;
    margin-left: auto;
    white-space: nowrap;
}
.summary-card {
    background: #141414;
    border: 1px solid #262626;
    border-radius: 10px;
    padding: 20px;
    margin-top: 8px;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 16px;
    margin-top: 12px;
}
.stat-box {
    text-align: center;
    padding: 14px 8px;
    background: #1a1a1a;
    border-radius: 8px;
    border: 1px solid #262626;
}
.stat-value {
    font-size: 1.5em;
    font-weight: 700;
    color: #f4f4f5;
    line-height: 1.2;
}
.stat-label {
    font-size: 0.78em;
    color: #71717a;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.chunk-card {
    background: #141414;
    border: 1px solid #262626;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 10px;
}
.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #262626;
}
.chunk-rank {
    font-weight: 700;
    font-size: 0.85em;
    color: #e4e4e7;
}
.chunk-meta {
    font-size: 0.8em;
    color: #71717a;
}
.chunk-score {
    background: #0a1a0e;
    color: #4ade80;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.78em;
    font-weight: 600;
    border: 1px solid #166534;
}
.chunk-text {
    font-size: 0.88em;
    line-height: 1.6;
    color: #d4d4d8;
}
.answer-card {
    background: #141414;
    border: 1px solid #262626;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 14px;
}
.answer-card .answer-header {
    font-size: 0.82em;
    font-weight: 600;
    color: #71717a;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
}
.answer-card .answer-body {
    font-size: 0.92em;
    line-height: 1.7;
    color: #e4e4e7;
}
.answer-card .answer-body p { margin: 0 0 0.6em 0; }
.answer-card .answer-body p:last-child { margin-bottom: 0; }
.answer-card .answer-body ul, .answer-card .answer-body ol { margin: 0.4em 0; padding-left: 1.4em; }
.answer-card .answer-body code {
    background: #1a1a1a;
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 0.9em;
}
.answer-card .answer-body pre {
    background: #1a1a1a;
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 0.6em 0;
}
.answer-card .answer-body pre code { background: none; padding: 0; }
.sources-header {
    font-size: 0.82em;
    font-weight: 600;
    color: #71717a;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 18px 0 10px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #262626;
}
"""


# ── HTML helpers ──────────────────────────────────────────────────

def _step(label: str, state: str = "pending", detail: str = "") -> str:
    icons = {
        "pending": '<span class="step-icon">&#9675;</span>',
        "active":  '<span class="step-icon">&#8987;</span>',
        "done":    '<span class="step-icon">&#10003;</span>',
        "error":   '<span class="step-icon">&#10007;</span>',
    }
    detail_html = f'<span class="step-detail">{detail}</span>' if detail else ""
    return f'<div class="step step-{state}">{icons[state]}<span>{label}</span>{detail_html}</div>'


def _pipeline(*steps: str) -> str:
    return f'<div class="pipeline-steps">{"".join(steps)}</div>'


def _summary_card(stats: list[tuple[str, str]]) -> str:
    boxes = "".join(
        f'<div class="stat-box"><div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>'
        for val, label in stats
    )
    return f'<div class="summary-card"><div class="summary-grid">{boxes}</div></div>'


def _chunk_card(rank: int, text: str, file: str, chunk_idx: int, score: float) -> str:
    return (
        f'<div class="chunk-card">'
        f'  <div class="chunk-header">'
        f'    <span class="chunk-rank">#{rank}</span>'
        f'    <span class="chunk-meta">{file} &middot; chunk {chunk_idx}</span>'
        f'    <span class="chunk-score">{score:.4f}</span>'
        f'  </div>'
        f'  <div class="chunk-text">{text}</div>'
        f'</div>'
    )


# ── Ingest pipeline ──────────────────────────────────────────────

def run_ingest(files):
    """Process uploaded documents, yielding pipeline progress + summary."""
    if not files:
        raise gr.Error("Upload at least one file.")

    # -- initial state --
    yield (
        _pipeline(
            _step("Reading documents", "active"),
            _step("Splitting into chunks"),
            _step("Generating embeddings"),
            _step("Storing in database"),
        ),
        "",
    )

    # -- Step 1: Read documents --
    tmp_dir = tempfile.mkdtemp()
    extracted = {}   # name -> (path_on_disk, text)
    for f in files:
        if isinstance(f, str):
            path, name = f, os.path.basename(f)
        else:
            name = f.orig_name or os.path.basename(f.path)
            path = os.path.join(tmp_dir, name)
            shutil.copy2(f.path, path)
        try:
            text = file_to_text(path)
            if not text.strip():
                extracted[name] = None
            else:
                extracted[name] = (path, text)
        except Exception:
            extracted[name] = None

    usable = {k: v for k, v in extracted.items() if v is not None}
    failed = [k for k, v in extracted.items() if v is None]

    if not usable:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        yield (
            _pipeline(
                _step("Reading documents", "error", f"0/{len(files)} readable"),
                _step("Splitting into chunks"),
                _step("Generating embeddings"),
                _step("Storing in database"),
            ),
            "",
        )
        raise gr.Error("No readable text found. Try text-based PDF, .txt, or .docx files.")

    total_chars = sum(len(v[1]) for v in usable.values())
    total_words = sum(len(v[1].split()) for v in usable.values())
    read_detail = f"{len(usable)} file{'s' if len(usable) != 1 else ''}"
    if failed:
        read_detail += f" ({len(failed)} failed)"

    yield (
        _pipeline(
            _step("Reading documents", "done", read_detail),
            _step("Splitting into chunks", "active"),
            _step("Generating embeddings"),
            _step("Storing in database"),
        ),
        "",
    )

    # -- Steps 2 & 3: Chunk + Embed (with cache) --
    conn = get_db()
    all_chunks: list[dict] = []
    all_embeddings: list[list[float]] = []
    cached_count = 0

    for name, (path, text) in usable.items():
        fhash = file_hash(path)
        if is_cached(conn, fhash):
            cached = load_chunks(conn, fhash)
            for c in cached:
                all_chunks.append({"text": c["text"], "file": name, "chunk_index": c["chunk_index"]})
                all_embeddings.append(c["embedding"])
            cached_count += 1
        else:
            chunks = semantic_chunk(text)
            if chunks:
                embs = embed_texts(chunks)
                save_chunks(conn, fhash, path, chunks, embs)
                for i, (c, e) in enumerate(zip(chunks, embs)):
                    all_chunks.append({"text": c, "file": name, "chunk_index": i})
                    all_embeddings.append(e)

    conn.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    dim = len(all_embeddings[0]) if all_embeddings else 0
    cache_detail = f"{len(all_chunks)} chunks"
    if cached_count:
        cache_detail += f" ({cached_count} cached)"

    yield (
        _pipeline(
            _step("Reading documents", "done", read_detail),
            _step("Splitting into chunks", "done", cache_detail),
            _step("Generating embeddings", "done", f"dim {dim}"),
            _step("Storing in database", "active"),
        ),
        "",
    )

    # -- Step 4: Store --
    client = get_client()
    ensure_collection(client)
    count = upsert_points(all_chunks, all_embeddings, client=client)

    yield (
        _pipeline(
            _step("Reading documents", "done", read_detail),
            _step("Splitting into chunks", "done", f"{len(all_chunks)} chunks"),
            _step("Generating embeddings", "done", f"dim {dim}"),
            _step("Storing in database", "done", f"{count} stored"),
        ),
        _summary_card([
            (str(len(usable)), "Documents"),
            (f"{total_chars:,}", "Characters"),
            (f"{total_words:,}", "Words"),
            (str(len(all_chunks)), "Chunks"),
            (str(count), "Stored"),
        ]),
    )


# ── Query pipeline ───────────────────────────────────────────────

def run_query(question: str):
    if not question.strip():
        raise gr.Error("Enter a question.")

    top_k = 3

    # -- initial state --
    yield (
        _pipeline(
            _step("Embedding question", "active"),
            _step("Finding candidates"),
            _step("Reranking matches"),
            _step("Generating answer"),
        ),
        "",
    )

    # Step 1 — embed the query
    q_emb = embed_texts([question])[0]

    yield (
        _pipeline(
            _step("Embedding question", "done", f"dim {len(q_emb)}"),
            _step("Finding candidates", "active"),
            _step("Reranking matches"),
            _step("Generating answer"),
        ),
        "",
    )

    # Step 2 — vector search
    client = get_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb,
        limit=30,
        with_payload=True,
    ).points
    candidates = [{**p.payload, "score": p.score} for p in results]

    yield (
        _pipeline(
            _step("Embedding question", "done", f"dim {len(q_emb)}"),
            _step("Finding candidates", "done", f"{len(candidates)} found"),
            _step("Reranking matches", "active"),
            _step("Generating answer"),
        ),
        "",
    )

    # Step 3 — rerank
    reranked = rerank(question, candidates, top_k=top_k)

    yield (
        _pipeline(
            _step("Embedding question", "done", f"dim {len(q_emb)}"),
            _step("Finding candidates", "done", f"{len(candidates)} found"),
            _step("Reranking matches", "done", f"top {len(reranked)} selected"),
            _step("Generating answer", "active", "qwen3:1.7b"),
        ),
        "",
    )

    # Build sources HTML (available immediately after reranking)
    sources_header = '<div class="sources-header">Sources</div>'
    chunks_html = "".join(
        _chunk_card(
            rank=i,
            text=html_mod.escape(r["text"]),
            file=html_mod.escape(r.get("file", "?")),
            chunk_idx=r.get("chunk_index", 0),
            score=r["score"],
        )
        for i, r in enumerate(reranked, 1)
    )

    pipeline_generating = _pipeline(
        _step("Embedding question", "done", f"dim {len(q_emb)}"),
        _step("Finding candidates", "done", f"{len(candidates)} found"),
        _step("Reranking matches", "done", f"top {len(reranked)} selected"),
        _step("Generating answer", "active", "qwen3:1.7b"),
    )

    # Step 4 — stream answer token by token
    context = reranked[0]["text"] if reranked else ""
    md = MarkdownIt()

    if not context:
        answer = "No relevant context found."
    else:
        answer = ""
        for token in generate_answer_stream(question, context):
            answer += token
            answer_html = (
                '<div class="answer-card">'
                '  <div class="answer-header">Answer</div>'
                f'  <div class="answer-body">{md.render(answer)}</div>'
                '</div>'
            )
            yield (pipeline_generating, answer_html + sources_header + chunks_html)

    answer_html = (
        '<div class="answer-card">'
        '  <div class="answer-header">Answer</div>'
        f'  <div class="answer-body">{md.render(answer)}</div>'
        '</div>'
    )

    yield (
        _pipeline(
            _step("Embedding question", "done", f"dim {len(q_emb)}"),
            _step("Finding candidates", "done", f"{len(candidates)} found"),
            _step("Reranking matches", "done", f"top {len(reranked)} selected"),
            _step("Generating answer", "done", "qwen3:1.7b"),
        ),
        answer_html + sources_header + chunks_html,
    )


# ── Gradio UI ─────────────────────────────────────────────────────

with gr.Blocks(title="chat-local-docs", css=CSS) as demo:
    gr.Markdown("chat-local-docs", elem_classes=["main-title"])
    gr.Markdown("Search your documents using AI", elem_classes=["subtitle"])

    # ── TAB 1: IMPORT DOCUMENTS ──────────────────────────────────
    with gr.Tab("Import Documents"):
        file_input = gr.File(
            label="Documents",
            file_count="multiple",
            file_types=[".pdf", ".docx", ".txt"],
        )
        ingest_btn = gr.Button("Import", variant="primary")
        ingest_progress = gr.HTML(value="")
        ingest_summary = gr.HTML(value="")

        ingest_btn.click(
            fn=run_ingest,
            inputs=[file_input],
            outputs=[ingest_progress, ingest_summary],
        )

    # ── TAB 2: SEARCH ────────────────────────────────────────────
    with gr.Tab("Search"):
        question_input = gr.Textbox(
            label="Question",
            placeholder="e.g. What is the refund policy?",
        )
        query_btn = gr.Button("Search", variant="primary")
        query_progress = gr.HTML(value="")
        query_results = gr.HTML(value="")

        query_btn.click(
            fn=run_query,
            inputs=[question_input],
            outputs=[query_progress, query_results],
        )

if __name__ == "__main__":
    warmup_reranker()
    demo.launch(theme=theme)
