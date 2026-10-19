"""Evaluate retrieval and refusal behaviour on the sample handbook.

    python eval/run_eval.py              # full run (embeddings + LLM), prints a markdown table
    python eval/run_eval.py --sweep      # retrieval only: how each MIN_SCORE would behave

Each run indexes the sample into a throwaway Chroma directory, so it never touches your data.
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.llm import make_answer_model, make_embedder  # noqa: E402
from app.rag import RagService  # noqa: E402
from app.store import VectorStore  # noqa: E402

SAMPLE = ROOT / "sample" / "acme_handbook.pdf"
QUESTIONS = Path(__file__).with_name("questions.json")
RESULTS = Path(__file__).with_name("results.md")


@dataclass
class Row:
    question: str
    answerable: bool
    answered: bool
    page_hit: bool | None
    fact_ok: bool | None
    top_score: float
    latency_ms: int


def build_service(chroma_dir: str, min_score: float | None) -> RagService:
    settings = get_settings().model_copy(update={"chroma_dir": chroma_dir})
    if min_score is not None:
        settings = settings.model_copy(update={"min_score": min_score})
    return RagService(
        store=VectorStore(chroma_dir, "eval"),
        embedder=make_embedder(settings),
        llm=make_answer_model(settings),
        settings=settings,
    )


def load_questions() -> list[dict]:
    data = json.loads(QUESTIONS.read_text())
    return [{**q, "answerable": True} for q in data["answerable"]] + [
        {**q, "answerable": False} for q in data["unanswerable"]
    ]


def run(service: RagService, doc_id: str, questions: list[dict]) -> list[Row]:
    rows = []
    for q in questions:
        started = time.perf_counter()
        result = service.ask(doc_id, q["question"])
        latency = round((time.perf_counter() - started) * 1000)
        pages = {s.page for s in result.sources}
        rows.append(
            Row(
                question=q["question"],
                answerable=q["answerable"],
                answered=result.answered,
                page_hit=q["page"] in pages if q["answerable"] else None,
                fact_ok=any(e.lower() in result.answer.lower() for e in q["expect"]) if q["answerable"] else None,
                top_score=max((s.score for s in result.sources), default=0.0),
                latency_ms=latency,
            )
        )
        print(f"  {'✓' if result.answered == q['answerable'] else '✗'} {q['question']}", file=sys.stderr)
    return rows


def mark(value: bool | None) -> str:
    return "–" if value is None else ("✅" if value else "❌")


def report(rows: list[Row], min_score: float, model: str) -> str:
    answerable = [r for r in rows if r.answerable]
    unanswerable = [r for r in rows if not r.answerable]
    pct = lambda n, d: f"{100 * n / d:.0f}% ({n}/{d})"  # noqa: E731

    lines = [
        f"Model: `{model}` · MIN_SCORE = {min_score} · top_k = {get_settings().top_k}",
        "",
        "| # | Question | Type | Answered | Correct page in sources | Fact in answer | Top score | Latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        kind = "answerable" if r.answerable else "unanswerable"
        answered = mark(r.answered) if r.answerable else ("refused ✅" if not r.answered else "answered ❌")
        lines.append(
            f"| {i} | {r.question} | {kind} | {answered} | {mark(r.page_hit)} | {mark(r.fact_ok)} "
            f"| {r.top_score:.3f} | {r.latency_ms} ms |"
        )
    lines += [
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Retrieval hit rate (expected page in sources) | "
        f"{pct(sum(r.page_hit for r in answerable), len(answerable))} |",
        f"| Answerable questions answered | {pct(sum(r.answered for r in answerable), len(answerable))} |",
        f"| Answers containing the expected fact | {pct(sum(r.fact_ok for r in answerable), len(answerable))} |",
        f"| Unanswerable questions correctly refused | "
        f"{pct(sum(not r.answered for r in unanswerable), len(unanswerable))} |",
        f"| Median latency | {statistics.median(r.latency_ms for r in rows):.0f} ms |",
    ]
    return "\n".join(lines)


def sweep(service: RagService, doc_id: str, questions: list[dict]) -> str:
    """Retrieval-only threshold sweep: no LLM calls, just embeddings."""
    top = [(q["answerable"], service.retrieve(doc_id, q["question"])[0].score) for q in questions]
    answerable = sorted(s for a, s in top if a)
    unanswerable = sorted(s for a, s in top if not a)
    lines = [
        f"Top-1 scores, answerable:   min {answerable[0]:.3f} · max {answerable[-1]:.3f}",
        f"Top-1 scores, unanswerable: min {unanswerable[0]:.3f} · max {unanswerable[-1]:.3f}",
        "",
        "| MIN_SCORE | Answerable passed to LLM | Unanswerable stopped at layer 1 |",
        "|---|---|---|",
    ]
    for t in [round(0.20 + 0.025 * i, 3) for i in range(13)]:
        kept = sum(s >= t for s in answerable)
        blocked = sum(s < t for s in unanswerable)
        lines.append(f"| {t:.3f} | {kept}/{len(answerable)} | {blocked}/{len(unanswerable)} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="store_true", help="retrieval-only MIN_SCORE sweep (no LLM calls)")
    parser.add_argument("--min-score", type=float, help="override MIN_SCORE for this run")
    args = parser.parse_args()

    questions = load_questions()
    with tempfile.TemporaryDirectory() as chroma_dir:
        service = build_service(chroma_dir, args.min_score)
        doc = service.ingest(SAMPLE.name, SAMPLE.read_bytes())
        print(f"Indexed {doc.filename}: {doc.pages} pages, {doc.chunks} chunks", file=sys.stderr)

        if args.sweep:
            print(sweep(service, doc.doc_id, questions))
            return

        settings = service.settings
        model = settings.anthropic_model if settings.llm_provider == "anthropic" else settings.openai_chat_model
        output = report(run(service, doc.doc_id, questions), settings.min_score, model)
        RESULTS.write_text(output + "\n")
        print(output)
        print(f"\nSaved to {RESULTS.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
