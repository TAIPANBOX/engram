"""Tests for access logging, decay job, and importance integration."""

from datetime import UTC, datetime, timedelta

import pytest

from engram import DecayConfig, Engram
from engram.decay import run_decay


@pytest.fixture()
def mem() -> Engram:
    with Engram(path=":memory:") as m:
        yield m


# ------------------------------------------------------------------
# Access logging
# ------------------------------------------------------------------


def test_recall_logs_access(mem: Engram) -> None:
    mem.observe("The board approved the merger")
    mem.recall("merger approval")
    count, last = mem._store.get_access_stats(
        mem._store._conn.execute("SELECT id FROM episodes LIMIT 1").fetchone()[0]
    )
    assert count == 1
    assert last is not None


def test_recall_logs_rank(mem: Engram) -> None:
    for i in range(3):
        mem.observe(f"Meeting update number {i}")
    mem.recall("meeting", k=3)
    row = mem._store._conn.execute(
        "SELECT rank FROM access_log ORDER BY rank ASC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row[0] == 0  # top result has rank 0


def test_get_access_stats_empty(mem: Engram) -> None:
    mem.observe("Some event")
    ep_id = mem._store._conn.execute("SELECT id FROM episodes LIMIT 1").fetchone()[0]
    count, last = mem._store.get_access_stats(ep_id)
    assert count == 0
    assert last is None


def test_multiple_recalls_accumulate(mem: Engram) -> None:
    mem.observe("Alice presented the roadmap")
    ep_id = mem._store._conn.execute("SELECT id FROM episodes LIMIT 1").fetchone()[0]

    mem.recall("roadmap presentation")
    mem.recall("roadmap presentation")
    mem.recall("roadmap presentation")

    count, _ = mem._store.get_access_stats(ep_id)
    assert count == 3


# ------------------------------------------------------------------
# Decay job
# ------------------------------------------------------------------


def test_decay_updates_all_episodes(mem: Engram) -> None:
    for i in range(5):
        mem.observe(f"Event {i}")
    updated = mem.decay()
    assert updated == 5


def test_decay_reduces_score_for_old_episodes(mem: Engram) -> None:
    mem.observe("Ancient history event")
    ep_id = mem._store._conn.execute("SELECT id FROM episodes LIMIT 1").fetchone()[0]

    # Simulate 30 days ago by running decay with a future `now`
    future_now = datetime.now(tz=UTC) + timedelta(days=30)
    run_decay(mem._store, DecayConfig(), now=future_now)

    ep = mem._store.get_episode(ep_id)
    assert ep is not None
    assert ep.importance_score < 1.0  # decayed below initial default


def test_decay_reinforces_frequently_accessed(mem: Engram) -> None:
    mem.observe("Highly relevant fact")
    # Access many times
    for _ in range(20):
        mem.recall("highly relevant fact", k=1)
    # Decay with 10 days elapsed
    future_now = datetime.now(tz=UTC) + timedelta(days=10)
    run_decay(mem._store, DecayConfig(), now=future_now)
    ep_id = mem._store._conn.execute("SELECT id FROM episodes LIMIT 1").fetchone()[0]
    ep = mem._store.get_episode(ep_id)
    assert ep is not None
    # access reinforcement should keep score meaningfully positive
    assert ep.importance_score > 0.0


# ------------------------------------------------------------------
# SearchResult.importance
# ------------------------------------------------------------------


def test_search_result_exposes_importance(mem: Engram) -> None:
    mem.observe("Important announcement from the CEO")
    results = mem.recall("CEO announcement")
    assert len(results) == 1
    assert isinstance(results[0].importance, float)


def test_importance_reflects_stored_score(mem: Engram) -> None:
    mem.observe("Test event for score verification")
    ep_id = mem._store._conn.execute("SELECT id FROM episodes LIMIT 1").fetchone()[0]
    # Manually update importance
    mem._store.update_importance(ep_id, 0.42)
    results = mem.recall("test event")
    assert len(results) == 1
    assert abs(results[0].importance - 0.42) < 1e-6


# ------------------------------------------------------------------
# Engram.decay() public API
# ------------------------------------------------------------------


def test_engram_decay_returns_count(mem: Engram) -> None:
    for i in range(4):
        mem.observe(f"Episode {i}")
    assert mem.decay() == 4


def test_engram_decay_with_custom_config() -> None:
    cfg = DecayConfig(lambda_=0.5, alpha=0.3, beta=0.05)
    with Engram(path=":memory:", decay_config=cfg) as mem:
        mem.observe("Some event")
        updated = mem.decay()
        assert updated == 1
