"""Scaling benchmark: what recall costs as the store grows.

The latency suite answers "how fast is one call on a small store", which for
a memory you keep for months is the wrong question. This one sweeps the store
size and reports where exact search stops being free.

Vector search here is an exact scan: sqlite-vec compares the query against
every vector in the partition, so cost is linear in the number of episodes
that survive the filters. That is a deliberate trade (it is what makes
scoped, ``as_of`` and hard-delete semantics exact), and the point of this
benchmark is to say honestly where the trade starts to hurt.

Spreading activation is deliberately absent. Its graph edges are written by
``reflect()``, which needs an LLM, so on a store built by ``observe()`` alone
the graph is empty and the mode would degenerate into its seed KNN. Measuring
that and calling it spreading would flatter it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engram.benchmarks.latency import LatencyStats, _make_texts

if TYPE_CHECKING:
    from engram.core import Engram

# The lopsided shape that matters for scoped recall: one agent holds almost
# everything, another holds a handful. Before agent_id became a vec0 partition
# key this was also the shape that returned nothing at all.
_SOLO_EPISODES = 5

_BATCH = 500


@dataclass
class ScalePoint:
    """One store size and everything measured on it."""

    n_episodes: int
    file_mb: float
    rss_mb: float | None
    """Resident memory of a fresh process that opens this store and recalls
    from it, which is what an agent pays to use an existing memory."""
    build_s: float
    latencies: dict[str, LatencyStats] = field(default_factory=dict)


def _current_rss_mb() -> float | None:
    """Resident set size of this process, or None where it cannot be read.

    Deliberately not ``resource.getrusage``: that reports the *peak*, which
    never falls, so a sweep in one process would report the largest store's
    footprint for every size measured after it.
    """
    if sys.platform == "linux":
        try:
            with Path("/proc/self/statm").open() as fh:
                pages = int(fh.read().split()[1])
            return pages * 4096 / 1e6
        except (OSError, IndexError, ValueError):
            return None
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                capture_output=True,
                text=True,
                check=False,
            )
            return float(out.stdout.strip()) / 1000 if out.stdout.strip() else None
        except (OSError, ValueError):
            return None
    return None


_RSS_PROBE = """
import sys
from engram.core import Engram
from engram.benchmarks.scale import _current_rss_mb, _make_queries

mem = Engram(path=sys.argv[1])
for q in _make_queries(20):
    mem.recall(q, k=10, cross_agent=True)
print(_current_rss_mb())
"""


def _recall_rss_mb(path: str) -> float | None:
    """Resident memory of a fresh process that opens *path* and recalls from it.

    Measured in a child rather than in this process because building a store
    runs the embedder over every episode in batches, and those allocations are
    not returned to the OS. Reading RSS after a build reports the writer's high
    water mark, which at 500 episodes came to 436 MB against a 1.4 MB file. The
    number worth publishing is what an agent pays to *use* an existing memory.
    """
    out = subprocess.run(
        [sys.executable, "-c", _RSS_PROBE, path],
        capture_output=True,
        text=True,
        check=False,
    )
    line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    try:
        return float(line)
    except ValueError:
        return None


def _make_queries(n: int) -> list[str]:
    """Return *n* distinct query strings that appear nowhere in the corpus.

    Querying with corpus strings makes the embedder's LRU cache serve them for
    free while the corpus is small enough to fit, then miss once it is not.
    That reads as a scaling cliff at the size where eviction starts, and it is
    a property of the cache, not of the store. Distinct unseen strings make
    every measured call pay the embedding once, exactly as a real query does.
    """
    subjects = ["the migration", "the outage", "the budget", "the hiring round", "the audit"]
    places = ["in Q3", "last winter", "at the offsite", "before the merger", "during onboarding"]
    return [
        f"what was decided about {subjects[i % len(subjects)]} "
        f"{places[(i // len(subjects)) % len(places)]}, note {i}"
        for i in range(n)
    ]


def _build_store(path: str, n: int) -> tuple[Engram, Engram]:
    """Fill a fresh store with *n* episodes split across a bulk and a solo agent."""
    from engram.core import Engram
    from engram.models import ObserveInput

    bulk = Engram(path=path, agent_id="bulk")
    solo = Engram(path=path, agent_id="solo")

    texts = _make_texts(n)
    bulk_texts = texts[: n - _SOLO_EPISODES]
    for start in range(0, len(bulk_texts), _BATCH):
        bulk.observe_many([ObserveInput(content=t) for t in bulk_texts[start : start + _BATCH]])
    solo.observe_many([ObserveInput(content=t) for t in texts[n - _SOLO_EPISODES :]])
    return bulk, solo


def _time_calls(call: Callable[[str], Any], queries: list[str], warmup: int) -> LatencyStats:
    for q in queries[:warmup]:
        call(q)
    samples_ms: list[float] = []
    for q in queries[warmup:]:
        t0 = time.perf_counter()
        call(q)
        samples_ms.append((time.perf_counter() - t0) * 1000)
    return LatencyStats.from_samples(samples_ms)


def bench_scale_point(
    n: int,
    k: int = 10,
    n_queries: int = 50,
    warmup: int = 5,
    workdir: str | None = None,
) -> ScalePoint:
    """Build a store of *n* episodes and measure recall across its query paths."""
    tmp = workdir or tempfile.mkdtemp(prefix="engram-scale-")
    path = str(Path(tmp) / f"scale-{n}.engram")
    Path(path).unlink(missing_ok=True)

    t0 = time.perf_counter()
    bulk, solo = _build_store(path, n)
    build_s = time.perf_counter() - t0

    queries = _make_queries(n_queries + warmup)
    now = datetime.now(tz=UTC)
    store = bulk._store

    # Search is timed against a pre-computed embedding, so the column shows the
    # part that grows with the store rather than the flat embedding cost that
    # dominates it at every size measured here. The end-to-end column keeps
    # that cost in, because it is what a caller of recall() actually waits for.
    vectors = {q: bulk._embedder.embed(q) for q in queries}

    latencies = {
        "cosine": _time_calls(
            lambda q: store.search_episodes(vectors[q], k, agent_id=None), queries, warmup
        ),
        "hybrid": _time_calls(
            lambda q: store.search_episodes_hybrid(q, vectors[q], k, agent_id=None),
            queries,
            warmup,
        ),
        "as_of": _time_calls(
            lambda q: store.search_episodes_as_of(vectors[q], k, now, agent_id=None),
            queries,
            warmup,
        ),
        "scoped (5 of n)": _time_calls(
            lambda q: store.search_episodes(vectors[q], k, agent_id="solo"), queries, warmup
        ),
        "recall() end to end": _time_calls(
            lambda q: bulk.recall(q, k=k, cross_agent=True),
            _make_queries(2 * (n_queries + warmup))[n_queries + warmup :],
            warmup,
        ),
    }

    # WAL holds recent writes outside the main file, so stat() on its own
    # reports a store that looks empty until SQLite happens to checkpoint.
    store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    file_mb = Path(path).stat().st_size / 1e6
    bulk.close()
    solo.close()

    point = ScalePoint(
        n_episodes=n,
        file_mb=file_mb,
        rss_mb=_recall_rss_mb(path),
        build_s=build_s,
        latencies=latencies,
    )
    if workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return point


def run_scale_suite(
    sizes: tuple[int, ...] = (1_000, 10_000, 100_000),
    k: int = 10,
    n_queries: int = 50,
) -> list[ScalePoint]:
    """Measure every size in *sizes*, smallest first."""
    return [bench_scale_point(n, k=k, n_queries=n_queries) for n in sorted(sizes)]
