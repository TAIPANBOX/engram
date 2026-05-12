"""Latency benchmark: measures observe() and recall() p50/p99 timing."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class LatencyStats:
    """Timing statistics for a single operation."""

    p50_ms: float
    p99_ms: float
    mean_ms: float
    throughput_per_sec: float

    def __str__(self) -> str:
        return (
            f"p50={self.p50_ms:.1f}ms  p99={self.p99_ms:.1f}ms  "
            f"mean={self.mean_ms:.1f}ms  tput={self.throughput_per_sec:.0f}/s"
        )


def _make_texts(n: int) -> list[str]:
    """Return n deterministic synthetic episode strings."""
    names = ["Alice", "Bob", "Ivan", "Maria", "Chen", "Fatima", "Luca", "Sara"]
    orgs = ["Acme", "Globex", "Initech", "Umbrella", "Hooli", "Pied Piper"]
    roles = ["engineer", "manager", "designer", "analyst", "researcher"]
    cities = ["Berlin", "Kyiv", "Paris", "London", "Tokyo", "Nairobi"]
    verbs = ["joined", "left", "visited", "presented at", "called", "emailed"]
    texts = []
    for i in range(n):
        name = names[i % len(names)]
        verb = verbs[i % len(verbs)]
        org = orgs[i % len(orgs)]
        city = cities[i % len(cities)]
        role = roles[i % len(roles)]
        texts.append(f"{name} {verb} {org} in {city} as a {role} (event {i}).")
    return texts


def bench_observe(n: int = 200, warmup: int = 5) -> LatencyStats:
    """Measure observe() latency over *n* calls after *warmup* discarded calls.

    Args:
        n: Number of measured calls.
        warmup: Calls to discard (allow model + SQLite warm-up).

    Returns:
        :class:`LatencyStats` for the observe operation.
    """
    from engram.core import Engram

    mem = Engram()
    texts = _make_texts(n + warmup)

    for i in range(warmup):
        mem.observe(texts[i])

    samples_ms: list[float] = []
    for i in range(warmup, warmup + n):
        t0 = time.perf_counter()
        mem.observe(texts[i])
        samples_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(samples_ms)
    total_s = sum(samples_ms) / 1000
    return LatencyStats(
        p50_ms=float(np.percentile(arr, 50)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(arr.mean()),
        throughput_per_sec=n / total_s if total_s > 0 else 0.0,
    )


def bench_recall(
    n_episodes: int = 200,
    n_queries: int = 100,
    k: int = 5,
    mode: str = "cosine",
    warmup: int = 5,
) -> LatencyStats:
    """Measure recall() latency over *n_queries* calls.

    Pre-populates the store with *n_episodes* episodes before measuring.

    Args:
        n_episodes: Episodes to pre-load.
        n_queries: Number of measured recall calls.
        k: Recall top-k.
        mode: ``"cosine"`` or ``"spreading"``.
        warmup: Queries to discard before measuring.

    Returns:
        :class:`LatencyStats` for the recall operation.
    """
    from engram.core import Engram

    mem = Engram()
    texts = _make_texts(n_episodes)
    for t in texts:
        mem.observe(t)

    queries = _make_texts(n_queries + warmup)

    for i in range(warmup):
        mem.recall(queries[i], k=k, mode=mode)

    samples_ms: list[float] = []
    for i in range(warmup, warmup + n_queries):
        t0 = time.perf_counter()
        mem.recall(queries[i], k=k, mode=mode)
        samples_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(samples_ms)
    total_s = sum(samples_ms) / 1000
    return LatencyStats(
        p50_ms=float(np.percentile(arr, 50)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(arr.mean()),
        throughput_per_sec=n_queries / total_s if total_s > 0 else 0.0,
    )


def run_latency_suite(n: int = 200, k: int = 5) -> dict[str, LatencyStats]:
    """Run the full latency suite and return results keyed by operation name."""
    return {
        "observe": bench_observe(n=n),
        "recall_cosine": bench_recall(n_episodes=n, n_queries=n // 2, k=k, mode="cosine"),
        "recall_spreading": bench_recall(n_episodes=n, n_queries=n // 2, k=k, mode="spreading"),
    }
