"""Tests for fact and reflection CRUD in Store."""

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from engram.models import Fact, ReflectionRun
from engram.schema import migrate
from engram.store import Store

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _make_store() -> Store:
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    return Store(conn, dim=384)


def _make_fact(
    subject: str = "Ivan",
    predicate: str = "works_at",
    obj: str = "Acme",
    fact_id: str = "f1",
    valid_from: datetime = _T0,
) -> Fact:
    return Fact(
        id=fact_id,
        subject=subject,
        predicate=predicate,
        object=obj,
        valid_from=valid_from,
        valid_to=None,
        recorded_at=valid_from,
        superseded_at=None,
        superseded_by=None,
        confidence=0.9,
        derived_from=["ep1", "ep2"],
        extracted_by=None,
    )


# ------------------------------------------------------------------
# Facts CRUD
# ------------------------------------------------------------------


def test_insert_and_get_fact() -> None:
    store = _make_store()
    fact = _make_fact()
    store.insert_fact(fact)
    retrieved = store.get_fact("f1")
    assert retrieved is not None
    assert retrieved.subject == "Ivan"
    assert retrieved.predicate == "works_at"
    assert retrieved.object == "Acme"
    assert retrieved.confidence == pytest.approx(0.9)
    assert retrieved.derived_from == ["ep1", "ep2"]
    assert retrieved.extracted_by is None


def test_get_fact_missing() -> None:
    store = _make_store()
    assert store.get_fact("nonexistent") is None


def test_get_active_facts_returns_only_open() -> None:
    store = _make_store()
    f1 = _make_fact(fact_id="f1", obj="Acme")
    f2 = _make_fact(fact_id="f2", obj="Globex")
    store.insert_fact(f1)
    store.insert_fact(f2)
    # Close f1
    store.close_fact("f1", valid_to=_T0 + timedelta(days=1), superseded_by="f2")
    active = store.get_active_facts("Ivan", "works_at")
    assert len(active) == 1
    assert active[0].id == "f2"


def test_close_fact_sets_fields() -> None:
    store = _make_store()
    store.insert_fact(_make_fact())
    t1 = _T0 + timedelta(days=5)
    store.close_fact("f1", valid_to=t1, superseded_by="f2")
    fact = store.get_fact("f1")
    assert fact is not None
    assert fact.valid_to is not None
    assert fact.superseded_by == "f2"
    assert fact.superseded_at is not None


def test_get_all_facts_includes_closed() -> None:
    store = _make_store()
    f1 = _make_fact(fact_id="f1", obj="Acme")
    f2 = _make_fact(fact_id="f2", obj="Globex")
    store.insert_fact(f1)
    store.insert_fact(f2)
    store.close_fact("f1", valid_to=_T0 + timedelta(days=1), superseded_by="f2")
    all_facts = store.get_all_facts("Ivan")
    assert len(all_facts) == 2


def test_get_all_active_facts() -> None:
    store = _make_store()
    store.insert_fact(_make_fact(fact_id="f1", subject="Ivan", predicate="works_at"))
    store.insert_fact(_make_fact(fact_id="f2", subject="Ivan", predicate="lives_in", obj="Berlin"))
    store.insert_fact(_make_fact(fact_id="f3", subject="Maria", predicate="works_at", obj="Acme"))
    active = store.get_all_active_facts()
    assert len(active) == 3
    store.close_fact("f1", valid_to=_T0 + timedelta(days=1), superseded_by="f3")
    active = store.get_all_active_facts()
    assert len(active) == 2


# ------------------------------------------------------------------
# Episodes since
# ------------------------------------------------------------------


def test_get_episodes_since_none_returns_all(populated_store: Store) -> None:
    episodes = populated_store.get_episodes_since(None)
    assert len(episodes) == 3


def test_get_episodes_since_filters_by_time(populated_store: Store) -> None:
    cutoff = _T0 + timedelta(hours=1, minutes=30)
    episodes = populated_store.get_episodes_since(cutoff)
    # Only the episode timestamped at _T0 + 2h should be after the cutoff
    assert len(episodes) == 1


@pytest.fixture()
def populated_store() -> Store:
    """Store with 3 episodes at T0, T0+1h, T0+2h."""
    import numpy as np

    from engram.models import Episode

    store = _make_store()
    for i in range(3):
        ts = _T0 + timedelta(hours=i)
        ep = Episode(id=f"ep{i}", content=f"Event {i}", timestamp=ts)
        vec = np.zeros(384, dtype=np.float32)
        store.insert_episode(ep, vec)
    return store


# ------------------------------------------------------------------
# Prune
# ------------------------------------------------------------------


def test_prune_removes_low_importance(populated_store: Store) -> None:
    # Set one episode to low importance
    populated_store.update_importance("ep0", 0.05)
    pruned = populated_store.prune_episodes(threshold=0.1)
    assert pruned == 1
    assert populated_store.episode_count() == 2


def test_prune_nothing_above_threshold(populated_store: Store) -> None:
    pruned = populated_store.prune_episodes(threshold=0.0)
    assert pruned == 0
    assert populated_store.episode_count() == 3


# ------------------------------------------------------------------
# Reflections CRUD
# ------------------------------------------------------------------


def test_insert_and_get_last_reflection() -> None:
    store = _make_store()
    run = ReflectionRun(id="r1", started_at=_T0, model_used="stub")
    store.insert_reflection(run)
    last = store.get_last_reflection()
    assert last is not None
    assert last.id == "r1"
    assert last.model_used == "stub"


def test_get_last_reflection_empty() -> None:
    store = _make_store()
    assert store.get_last_reflection() is None


def test_update_reflection() -> None:
    store = _make_store()
    store.insert_reflection(ReflectionRun(id="r1", started_at=_T0))
    t1 = _T0 + timedelta(seconds=5)
    store.update_reflection(
        "r1",
        finished_at=t1,
        episodes_processed=10,
        facts_extracted=3,
        contradictions_resolved=1,
        cost_tokens=500,
    )
    run = store.get_reflection_by_id("r1")
    assert run is not None
    assert run.finished_at is not None
    assert run.episodes_processed == 10
    assert run.facts_extracted == 3
    assert run.cost_tokens == 500


def test_get_last_reflection_returns_most_recent() -> None:
    store = _make_store()
    store.insert_reflection(ReflectionRun(id="r1", started_at=_T0))
    store.insert_reflection(ReflectionRun(id="r2", started_at=_T0 + timedelta(hours=1)))
    last = store.get_last_reflection()
    assert last is not None
    assert last.id == "r2"
