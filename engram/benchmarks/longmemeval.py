"""Retrieval evaluation on LongMemEval-S.

LongMemEval (Wu et al., ICLR 2025, MIT licensed) asks a question against a
chat history of 30 to 60 sessions and marks which sessions, and which
individual turns, hold the evidence. That makes retrieval measurable without
an LLM judge: load the history, ask the question, and check whether the
evidence came back inside the top k.

Two numbers, because the benchmark supports two granularities and they are
not interchangeable:

``session_recall@k``
    A hit if any episode in the top k came from a session listed in
    ``answer_session_ids``. This is the looser of the two: a session can run
    to a dozen turns, so hitting it does not mean the answer was retrieved.

``turn_recall@k``
    A hit if any episode in the top k is a turn the dataset flagged
    ``has_answer``. 896 turns out of 246 750 carry that flag, so this is a
    far harder target and the one that reflects what an agent would actually
    read.

Anyone quoting a single "R@k" for a memory system should be asked which of
these it is; the gap between them is large.

The dataset is not vendored: it is 265 MB. Fetch it from
https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned and pass the
path with ``--data``.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DATE_FORMATS = ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y/%m/%d")


@dataclass
class LongMemEvalResult:
    """Retrieval scores over a LongMemEval run."""

    n_questions: int
    n_episodes: int
    mode: str
    k_values: tuple[int, ...]
    sampled: bool = False
    session_recall: dict[int, float] = field(default_factory=dict)
    turn_recall: dict[int, float] = field(default_factory=dict)
    session_recall_by_type: dict[str, dict[int, float]] = field(default_factory=dict)
    ingest_s: float = 0.0
    query_s: float = 0.0


def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _ingest(mem: Any, question: dict[str, Any]) -> int:
    """Load one question's chat history, a turn per episode.

    The session id rides along as a tag, and the evidence flag as a second
    tag. Neither is embedded or full-text indexed (only ``content`` is), so
    tagging cannot leak the answer into what retrieval matches on.
    """
    from engram.models import ObserveInput

    session_ids = question["haystack_session_ids"]
    dates = question.get("haystack_dates") or []
    items: list[ObserveInput] = []
    for idx, session in enumerate(question["haystack_sessions"]):
        session_id = session_ids[idx]
        when = _parse_date(dates[idx] if idx < len(dates) else None)
        for turn in session:
            content = turn.get("content")
            if not content:
                continue
            tags = [session_id]
            if turn.get("has_answer"):
                tags.append("__evidence__")
            items.append(ObserveInput(content=content, tags=tags, timestamp=when))

    for start in range(0, len(items), 500):
        mem.observe_many(items[start : start + 500])
    return len(items)


def _stratified_sample(
    questions: list[dict[str, Any]], size: int, seed: int
) -> list[dict[str, Any]]:
    """Take *size* questions keeping each type's share of the whole.

    The dataset file is ordered by question type: the first 70 entries are all
    ``single-session-user``, where the answer sits in one user turn. Taking the
    first N therefore measures the easiest category and calls it a subsample.
    """
    by_type: dict[str, list[dict[str, Any]]] = {}
    for q in questions:
        by_type.setdefault(q["question_type"], []).append(q)

    rng = random.Random(seed)
    total = len(questions)
    picked: list[dict[str, Any]] = []
    for qtype in sorted(by_type):
        group = by_type[qtype]
        take = min(len(group), max(1, round(size * len(group) / total)))
        picked.extend(rng.sample(group, take))
    rng.shuffle(picked)
    return picked


@dataclass(frozen=True)
class Variant:
    """One retrieval configuration to score, and the label it reports under.

    Modes and blend weights are both just configurations of the same call, so
    they are scored the same way. Ingesting a question's history costs about
    40 seconds and querying it costs 12 milliseconds, which is why a sweep of
    ten configurations is essentially free while a second full run would not
    be.
    """

    label: str
    mode: str
    vector_weight: float | None = None
    fts_weight: float | None = None

    def kwargs(self) -> dict[str, float]:
        if self.vector_weight is None or self.fts_weight is None:
            return {}
        return {"vector_weight": self.vector_weight, "fts_weight": self.fts_weight}


def default_weight_sweep() -> tuple[Variant, ...]:
    """Cosine, then hybrid across the blend, including the shipped default.

    The 0.7 / 0.3 default was inherited, never measured. This sweep is what
    turns it into either a justified number or a corrected one. The endpoints
    matter too: hybrid at 1.0 / 0.0 is not the same as ``mode="cosine"``,
    because the candidate pool and the min-max normalisation still differ.
    """
    weights = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
    return (
        Variant(label="cosine", mode="cosine"),
        *(
            Variant(
                label=f"hybrid v{v:.1f}/f{1 - v:.1f}",
                mode="hybrid",
                vector_weight=v,
                fts_weight=round(1 - v, 3),
            )
            for v in weights
        ),
    )


def variants_from_modes(modes: tuple[str, ...]) -> tuple[Variant, ...]:
    """Plain mode names, each at its own default weights."""
    return tuple(Variant(label=m, mode=m) for m in modes)


def _score_question(
    question: dict[str, Any],
    results_by_label: dict[str, list[Any]],
    k_values: tuple[int, ...],
) -> dict[str, dict[str, dict[str, bool]]]:
    """Reduce one question's recall output to hit flags per variant and k."""
    gold_sessions = set(question.get("answer_session_ids") or [])
    hits: dict[str, dict[str, dict[str, bool]]] = {}
    for label, results in results_by_label.items():
        hits[label] = {"session": {}, "turn": {}}
        for k in k_values:
            top = results[:k]
            hits[label]["session"][str(k)] = any(gold_sessions & set(r.episode.tags) for r in top)
            hits[label]["turn"][str(k)] = any("__evidence__" in r.episode.tags for r in top)
    return hits


def _aggregate(
    records: list[dict[str, Any]],
    k_values: tuple[int, ...],
    variants: tuple[Variant, ...],
    sampled: bool,
) -> dict[str, LongMemEvalResult]:
    """Build per-variant results from per-question records.

    Aggregation reads only the records, so a run that died partway can be
    scored from its checkpoint without repeating a single embedding.
    """
    n = len(records)
    type_totals: dict[str, int] = {}
    for rec in records:
        type_totals[rec["question_type"]] = type_totals.get(rec["question_type"], 0) + 1

    out: dict[str, LongMemEvalResult] = {}
    for variant in variants:
        label = variant.label
        session_hits = dict.fromkeys(k_values, 0)
        turn_hits = dict.fromkeys(k_values, 0)
        by_type: dict[str, dict[int, int]] = {}
        for rec in records:
            hits = rec["hits"].get(label)
            if hits is None:
                continue
            by_type.setdefault(rec["question_type"], dict.fromkeys(k_values, 0))
            for k in k_values:
                if hits["session"].get(str(k)):
                    session_hits[k] += 1
                    by_type[rec["question_type"]][k] += 1
                if hits["turn"].get(str(k)):
                    turn_hits[k] += 1
        out[label] = LongMemEvalResult(
            sampled=sampled,
            n_questions=n,
            n_episodes=sum(r["n_episodes"] for r in records),
            mode=label,
            k_values=k_values,
            session_recall={k: session_hits[k] / n for k in k_values} if n else {},
            turn_recall={k: turn_hits[k] / n for k in k_values} if n else {},
            session_recall_by_type={
                t: {k: by_type[t][k] / type_totals[t] for k in k_values} for t in sorted(by_type)
            },
            ingest_s=sum(r["ingest_s"] for r in records),
            query_s=sum(r["query_s"].get(label, 0.0) for r in records),
        )
    return out


def load_checkpoint(path: str | Path) -> list[dict[str, Any]]:
    """Read per-question records, skipping a truncated final line.

    A run killed mid-write leaves a partial line; it is dropped rather than
    failing the load, since the question it describes was never scored.
    """
    file = Path(path)
    if not file.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def run_longmemeval(
    data_path: str | Path,
    k_values: tuple[int, ...] = (5, 10),
    modes: tuple[str, ...] = ("cosine",),
    variants: tuple[Variant, ...] | None = None,
    sample: int | None = None,
    seed: int = 0,
    progress: Any = None,
    checkpoint: str | Path | None = None,
    resume: bool = False,
) -> dict[str, LongMemEvalResult]:
    """Score retrieval over the questions in *data_path*, one result per variant.

    Each question gets its own in-memory store, which is the benchmark's own
    protocol: the histories are per-question and pooling them would let a
    question be answered from another question's haystack.

    Every variant is queried against the same freshly built store before it is
    dropped. Ingesting a question's history costs about 40 seconds and a query
    costs 12 milliseconds, so comparing ten configurations in one pass is
    nearly free while running the benchmark once per configuration would not
    be.

    Args:
        variants: Configurations to score. Defaults to one per entry in
            *modes*; pass :func:`default_weight_sweep` to measure the blend
            weights instead of assuming them.

    Args:
        sample: Evaluate a stratified subsample of this size instead of all
            500 questions, keeping each question type's share. The full run
            embeds 246 738 turns and takes hours on a laptop.
        seed: Sampling seed, so a subsample is reproducible.
        checkpoint: Append one JSON record per scored question here. The run
            takes six hours and holds nothing else durable, so without this a
            failure at question 300 discards every hour that preceded it.
        resume: Skip questions already present in the checkpoint and score the
            rest, then aggregate over both.
    """
    import os

    from engram.core import Engram
    from engram.embedder import Embedder

    if variants is None:
        variants = variants_from_modes(modes)

    questions: list[dict[str, Any]] = json.loads(Path(data_path).read_text(encoding="utf-8"))
    sampled = sample is not None and sample < len(questions)
    if sampled:
        questions = _stratified_sample(questions, sample or 0, seed)

    # One embedder for the whole run rather than one per question. It keeps the
    # ONNX session alive across 500 stores, and its cache catches the 23% of
    # turns that LongMemEval reuses between haystacks. Threads are raised
    # because this is a bulk load; the library default suits an agent embedding
    # one turn at a time and is left alone.
    embedder = Embedder(cache_size=50_000, threads=os.cpu_count())

    max_k = max(k_values)
    records: list[dict[str, Any]] = []
    done_ids: set[str] = set()
    if checkpoint is not None and resume:
        records = load_checkpoint(checkpoint)
        done_ids = {r["question_id"] for r in records}

    pending = [q for q in questions if q["question_id"] not in done_ids]
    total = len(questions)

    for n_done, question in enumerate(pending, start=len(done_ids) + 1):
        mem = Engram()
        mem._embedder = embedder
        try:
            t0 = time.perf_counter()
            n_ep = _ingest(mem, question)
            ingest = time.perf_counter() - t0

            results_by_label: dict[str, list[Any]] = {}
            query_s: dict[str, float] = {}
            for variant in variants:
                t0 = time.perf_counter()
                results_by_label[variant.label] = mem.recall(
                    question["question"], k=max_k, mode=variant.mode, **variant.kwargs()
                )
                query_s[variant.label] = time.perf_counter() - t0
        finally:
            mem.close()

        record = {
            "question_id": question["question_id"],
            "question_type": question["question_type"],
            "n_episodes": n_ep,
            "ingest_s": ingest,
            "query_s": query_s,
            "hits": _score_question(question, results_by_label, k_values),
        }
        records.append(record)

        if checkpoint is not None:
            # Flushed per question: the point of the file is to survive a kill,
            # and a buffered write would lose exactly what it exists to keep.
            with Path(checkpoint).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
                fh.flush()

        if progress is not None:
            progress(n_done, total)

    return _aggregate(records, k_values, variants, sampled)
