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


def run_longmemeval(
    data_path: str | Path,
    k_values: tuple[int, ...] = (5, 10),
    modes: tuple[str, ...] = ("cosine",),
    sample: int | None = None,
    seed: int = 0,
    progress: Any = None,
) -> dict[str, LongMemEvalResult]:
    """Score retrieval over the questions in *data_path*, one result per mode.

    Each question gets its own in-memory store, which is the benchmark's own
    protocol: the histories are per-question and pooling them would let a
    question be answered from another question's haystack.

    Every mode is queried against the same freshly built store before it is
    dropped. Ingesting a question's history costs about 45 seconds and a query
    costs 11 milliseconds, so comparing modes in one pass is nearly free while
    running the benchmark once per mode would not be.

    Args:
        sample: Evaluate a stratified subsample of this size instead of all
            500 questions, keeping each question type's share. The full run
            embeds 246 738 turns and takes hours on a laptop.
        seed: Sampling seed, so a subsample is reproducible.
    """
    import os

    from engram.core import Engram
    from engram.embedder import Embedder

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
    session_hits = {m: dict.fromkeys(k_values, 0) for m in modes}
    turn_hits = {m: dict.fromkeys(k_values, 0) for m in modes}
    by_type: dict[str, dict[str, dict[int, int]]] = {m: {} for m in modes}
    type_totals: dict[str, int] = {}
    n_episodes = 0
    ingest_s = 0.0
    query_s = dict.fromkeys(modes, 0.0)

    for n_done, question in enumerate(questions, start=1):
        gold_sessions = set(question.get("answer_session_ids") or [])
        qtype = question["question_type"]
        type_totals[qtype] = type_totals.get(qtype, 0) + 1

        mem = Engram()
        mem._embedder = embedder
        try:
            t0 = time.perf_counter()
            n_episodes += _ingest(mem, question)
            ingest_s += time.perf_counter() - t0

            for mode in modes:
                by_type[mode].setdefault(qtype, dict.fromkeys(k_values, 0))
                t0 = time.perf_counter()
                results = mem.recall(question["question"], k=max_k, mode=mode)
                query_s[mode] += time.perf_counter() - t0

                for k in k_values:
                    top = results[:k]
                    if any(gold_sessions & set(r.episode.tags) for r in top):
                        session_hits[mode][k] += 1
                        by_type[mode][qtype][k] += 1
                    if any("__evidence__" in r.episode.tags for r in top):
                        turn_hits[mode][k] += 1
        finally:
            mem.close()

        if progress is not None:
            progress(n_done, len(questions))

    n = len(questions)
    return {
        mode: LongMemEvalResult(
            sampled=sampled,
            n_questions=n,
            n_episodes=n_episodes,
            mode=mode,
            k_values=k_values,
            session_recall={k: session_hits[mode][k] / n for k in k_values} if n else {},
            turn_recall={k: turn_hits[mode][k] / n for k in k_values} if n else {},
            session_recall_by_type={
                t: {k: by_type[mode][t][k] / type_totals[t] for k in k_values}
                for t in sorted(by_type[mode])
            },
            ingest_s=ingest_s,
            query_s=query_s[mode],
        )
        for mode in modes
    }
