"""Recall accuracy harness for data in LoCoMo's format.

This measures whatever file it is given. The fixture bundled in ``data/`` is
synthetic, written by hand alongside the questions it answers, so its scores
are a smoke test for the retrieval path and not a result on any benchmark.
Reporting them as one would be a claim about a dataset that was never run.


Data format (JSON):
    {
      "sessions": [{"id": "s1", "turns": [{"speaker": "...", "text": "..."}]}],
      "questions": [{"id": "q1", "question": "...", "answer": "...", "keywords": [...]}]
    }

A question is considered a "hit" at rank k if any keyword appears
(case-insensitive) in any of the top-k recalled episode texts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoCoMoResult:
    """Evaluation metrics from a LoCoMo benchmark run."""

    n_sessions: int
    n_episodes: int
    n_questions: int
    hit_rate_at_1: float
    hit_rate_at_5: float
    mrr: float  # mean reciprocal rank (within top-5)

    def __str__(self) -> str:
        return (
            f"sessions={self.n_sessions}  episodes={self.n_episodes}  "
            f"questions={self.n_questions}\n"
            f"  hit@1={self.hit_rate_at_1:.1%}  "
            f"hit@5={self.hit_rate_at_5:.1%}  "
            f"MRR={self.mrr:.3f}"
        )


def _load_data(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    path = Path(source)
    with path.open() as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def _hit(episode_content: str, keywords: list[str]) -> bool:
    lower = episode_content.lower()
    return any(kw.lower() in lower for kw in keywords)


def run_locomo(
    source: str | Path | dict[str, Any],
    k: int = 5,
    engram_path: str = ":memory:",
) -> LoCoMoResult:
    """Ingest sessions from *source* into Engram, then evaluate QA recall.

    Args:
        source: Path to a LoCoMo JSON file, or a pre-loaded dict.
        k: Top-k episodes to retrieve per question (max evaluated rank).
        engram_path: Engram file path (``":memory:"`` by default).

    Returns:
        :class:`LoCoMoResult` with hit_rate and MRR metrics.
    """
    from engram.core import Engram

    data = _load_data(source)
    sessions: list[dict[str, Any]] = data["sessions"]
    questions: list[dict[str, Any]] = data["questions"]

    mem = Engram(path=engram_path)
    n_episodes = 0
    for session in sessions:
        for turn in session["turns"]:
            mem.observe(
                turn["text"],
                actors=[turn["speaker"]],
                tags=["conversation", session["id"]],
            )
            n_episodes += 1

    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks: list[float] = []

    for q in questions:
        results = mem.recall(q["question"], k=k)
        keywords: list[str] = q["keywords"]

        first_hit_rank: int | None = None
        for rank, r in enumerate(results, start=1):
            if _hit(r.episode.content, keywords) and first_hit_rank is None:
                first_hit_rank = rank
                if rank == 1:
                    hits_at_1 += 1
                hits_at_k += 1
                break

        reciprocal_ranks.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

    n_q = len(questions)
    return LoCoMoResult(
        n_sessions=len(sessions),
        n_episodes=n_episodes,
        n_questions=n_q,
        hit_rate_at_1=hits_at_1 / n_q if n_q else 0.0,
        hit_rate_at_5=hits_at_k / n_q if n_q else 0.0,
        mrr=sum(reciprocal_ranks) / n_q if n_q else 0.0,
    )
