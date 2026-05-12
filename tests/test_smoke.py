"""Smoke test: 1000 episodes — correctness and basic performance."""

import time

import pytest

from engram import Engram

TOPICS = [
    "quarterly earnings report",
    "product roadmap discussion",
    "customer support escalation",
    "engineering sprint planning",
    "security incident review",
]

NEEDLE = "Ivan Kovalenko accepted the CTO offer at Globex Corporation"


@pytest.fixture(scope="module")
def populated_mem() -> Engram:
    mem = Engram(path=":memory:")
    episodes = [f"{TOPICS[i % len(TOPICS)]} — item {i}" for i in range(999)]
    episodes.append(NEEDLE)

    start = time.monotonic()
    for text in episodes:
        mem.observe(text)
    elapsed = time.monotonic() - start

    assert elapsed < 120.0, f"1000 inserts took {elapsed:.1f}s — too slow"
    return mem


def test_smoke_episode_count(populated_mem: Engram) -> None:
    assert populated_mem._store.episode_count() == 1000


def test_smoke_recall_finds_needle(populated_mem: Engram) -> None:
    results = populated_mem.recall("Who became CTO at Globex?", k=5)
    assert len(results) > 0
    top = results[0].episode.content
    assert "Globex" in top or "CTO" in top


def test_smoke_recall_latency(populated_mem: Engram) -> None:
    start = time.monotonic()
    populated_mem.recall("sprint planning velocity", k=5)
    elapsed_ms = (time.monotonic() - start) * 1000
    assert elapsed_ms < 500, f"recall took {elapsed_ms:.1f}ms — too slow"
