"""RAG evaluation: auto-generate test sets and measure retrieval + answer quality."""

import json
import random
import re
import sys
from pathlib import Path

import ollama

from file_to_text import file_to_text, list_supported_files
from chunking import semantic_chunk
from embeddings import embed_texts
from vector_db import search
from reranking import rerank
from llm import generate_answer, MODEL

TESTSET_PATH = Path("eval/testset.json")
RESULTS_PATH = Path("eval/results.json")


# ── Test set generation ──────────────────────────────────────────


def _generate_qa(chunk: str) -> dict | None:
    """Ask the LLM to produce a question + answer from a chunk."""
    prompt = (
        "You are a test-set generator for a document QA system.\n\n"
        "Given the following text passage, generate:\n"
        "1. A specific, self-contained question that this passage answers.\n"
        "2. A concise reference answer based ONLY on the passage.\n\n"
        "Respond in exactly this format (no extra text):\n"
        "QUESTION: <your question>\n"
        "ANSWER: <your answer>\n\n"
        f"Passage:\n{chunk}"
    )
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.message.content

    q_match = re.search(r"QUESTION:\s*(.+?)(?:\n|$)", text)
    a_match = re.search(r"ANSWER:\s*(.+)", text, re.DOTALL)
    if not q_match or not a_match:
        return None

    return {
        "question": q_match.group(1).strip(),
        "reference_answer": a_match.group(1).strip(),
    }


def generate_testset(directory: str = "data/", n_questions: int = 20) -> None:
    """Generate a test set by sampling chunks and creating QA pairs."""
    files = list_supported_files(directory)
    if not files:
        print(f"No supported files found in {directory}")
        sys.exit(1)

    print(f"Reading {len(files)} files...")
    all_chunks: list[dict] = []
    for file_path in files:
        text = file_to_text(str(file_path))
        chunks = semantic_chunk(text)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "file": file_path.name})

    print(f"  {len(all_chunks)} total chunks across {len(files)} files")

    # Sample chunks spread across files
    n = min(n_questions, len(all_chunks))
    sampled = random.sample(all_chunks, n)

    print(f"Generating {n} QA pairs...")
    testset: list[dict] = []
    for i, chunk_info in enumerate(sampled, 1):
        qa = _generate_qa(chunk_info["text"])
        if qa is None:
            print(f"  [{i}/{n}] skipped (failed to parse)")
            continue
        testset.append(
            {
                "question": qa["question"],
                "reference_answer": qa["reference_answer"],
                "source_file": chunk_info["file"],
                "source_chunk": chunk_info["text"],
            }
        )
        print(f"  [{i}/{n}] {qa['question'][:70]}...")

    TESTSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    TESTSET_PATH.write_text(json.dumps(testset, indent=2))
    print(f"\nSaved {len(testset)} test cases to {TESTSET_PATH}")


# ── Evaluation ───────────────────────────────────────────────────


def _word_overlap(a: str, b: str) -> float:
    """Fraction of words in `a` that also appear in `b`."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


def _find_chunk_rank(source_chunk: str, results: list[dict], threshold: float = 0.8) -> int | None:
    """Return 1-based rank of the source chunk in results, or None if not found."""
    for i, r in enumerate(results, 1):
        if _word_overlap(source_chunk, r["text"]) >= threshold:
            return i
    return None


def _judge_answer(question: str, answer: str, context: str) -> dict:
    """Use LLM-as-judge to score faithfulness and relevance (1-5)."""
    prompt = (
        "You are evaluating a QA system. Score the answer on two criteria.\n\n"
        f"Question: {question}\n\n"
        f"Context provided to the system:\n{context}\n\n"
        f"System's answer:\n{answer}\n\n"
        "Score each criterion from 1 (worst) to 5 (best):\n"
        "- FAITHFULNESS: Is the answer supported by the context? (1=hallucinated, 5=fully grounded)\n"
        "- RELEVANCE: Does the answer address the question? (1=off-topic, 5=directly answers)\n\n"
        "Respond in exactly this format:\n"
        "FAITHFULNESS: <1-5>\n"
        "RELEVANCE: <1-5>"
    )
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.message.content

    faithfulness = 3
    relevance = 3
    f_match = re.search(r"FAITHFULNESS:\s*(\d)", text)
    r_match = re.search(r"RELEVANCE:\s*(\d)", text)
    if f_match:
        faithfulness = int(f_match.group(1))
    if r_match:
        relevance = int(r_match.group(1))

    return {"faithfulness": faithfulness, "relevance": relevance}


def _load_results() -> list[dict]:
    """Load previously saved per-case results."""
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return []


def _save_results(results: list[dict]) -> None:
    """Persist per-case results to disk."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))


def _print_summary(results: list[dict], n_total: int) -> None:
    """Print aggregate metrics from saved results."""
    n = len(results)
    hits = {1: 0, 3: 0, 5: 0}
    mrr_sum = 0.0
    faithfulness_sum = 0.0
    relevance_sum = 0.0

    for r in results:
        rank = r["rank"]
        for k in hits:
            if rank is not None and rank <= k:
                hits[k] += 1
        if rank is not None:
            mrr_sum += 1.0 / rank
        faithfulness_sum += r["faithfulness"]
        relevance_sum += r["relevance"]

    print("\n" + "=" * 50)
    print(f"RESULTS ({n}/{n_total} cases evaluated)")
    print()
    print("RETRIEVAL")
    print(f"  Hit@1:  {hits[1]/n:.1%}  ({hits[1]}/{n})")
    print(f"  Hit@3:  {hits[3]/n:.1%}  ({hits[3]}/{n})")
    print(f"  Hit@5:  {hits[5]/n:.1%}  ({hits[5]}/{n})")
    print(f"  MRR:    {mrr_sum/n:.3f}")
    print()
    print("ANSWER QUALITY (1-5)")
    print(f"  Faithfulness: {faithfulness_sum/n:.2f}")
    print(f"  Relevance:    {relevance_sum/n:.2f}")
    print("=" * 50)
    print(f"\nPer-case results saved to {RESULTS_PATH}")


def run_eval(testset_path: str = str(TESTSET_PATH)) -> None:
    """Run retrieval + answer quality evaluation. Resumes from saved results."""
    path = Path(testset_path)
    if not path.exists():
        print(f"Test set not found at {path}. Run 'eval.py generate' first.")
        sys.exit(1)

    testset = json.loads(path.read_text())
    n = len(testset)

    # Resume: load existing results and skip already-evaluated questions
    results = _load_results()
    done_questions = {r["question"] for r in results}
    remaining = [(i, case) for i, case in enumerate(testset) if case["question"] not in done_questions]

    if results:
        print(f"Resuming: {len(results)}/{n} already evaluated, {len(remaining)} remaining\n")
    else:
        print(f"Evaluating {n} test cases...\n")

    for idx, (i, case) in enumerate(remaining, 1):
        question = case["question"]
        source_chunk = case["source_chunk"]
        print(f"[{len(results) + idx}/{n}] {question[:70]}...")

        # Retrieval
        candidates = search(question, top_k=30)
        reranked = rerank(question, candidates, top_k=5)

        rank = _find_chunk_rank(source_chunk, reranked)

        # Answer generation + judging
        context = reranked[0]["text"] if reranked else ""
        answer = generate_answer(question, context) if context else ""
        scores = _judge_answer(question, answer, context)

        result = {
            "question": question,
            "source_file": case["source_file"],
            "rank": rank,
            "faithfulness": scores["faithfulness"],
            "relevance": scores["relevance"],
            "answer": answer,
        }
        results.append(result)
        _save_results(results)

        status = f"rank={rank}" if rank else "miss"
        print(f"        {status}  faith={scores['faithfulness']}  rel={scores['relevance']}")

    _print_summary(results, n)


# ── CLI ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python eval.py generate [directory] [n_questions]")
        print("  python eval.py run [testset_path]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "generate":
        directory = sys.argv[2] if len(sys.argv) > 2 else "data/"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        generate_testset(directory, n)
    elif command == "run":
        testset_path = sys.argv[2] if len(sys.argv) > 2 else str(TESTSET_PATH)
        run_eval(testset_path)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
