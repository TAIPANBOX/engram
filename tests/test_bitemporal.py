"""Tests for bitemporal queries: as_of recall and timeline API."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from engram.models import Episode, Fact
from engram.schema import migrate
from engram.store import Store

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(days=30)
_T2 = _T0 + timedelta(days=60)
_T3 = _T0 + timedelta(days=90)


def _make_store() -> Store:
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    return Store(conn, dim=384)


def _insert_ep(store: Store, ep_id: str, ts: datetime, content: str = "text") -> None:
    ep = Episode(id=ep_id, content=content, timestamp=ts)
    vec = np.zeros(384, dtype=np.float32)
    store.insert_episode(ep, vec)


def _make_fact(
    store: Store,
    fact_id: str,
    subject: str,
    predicate: str,
    obj: str,
    valid_from: datetime,
    valid_to: datetime | None = None,
    superseded_by: str | None = None,
) -> Fact:
    fact = Fact(
        id=fact_id,
        subject=subject,
        predicate=predicate,
        object=obj,
        valid_from=valid_from,
        valid_to=valid_to,
        recorded_at=valid_from,
        superseded_at=valid_to,
        superseded_by=superseded_by,
        confidence=0.9,
    )
    store.insert_fact(fact)
    return fact


# ------------------------------------------------------------------
# search_episodes_as_of
# ------------------------------------------------------------------


def test_search_as_of_excludes_future_episodes() -> None:
    store = _make_store()
    _insert_ep(store, "ep_past", _T0, "old memory")
    _insert_ep(store, "ep_future", _T2, "new memory")

    vec = np.zeros(384, dtype=np.float32)
    results = store.search_episodes_as_of(vec, k=5, as_of=_T1)
    ids = {ep.id for ep, _, _ in results}
    assert "ep_past" in ids
    assert "ep_future" not in ids


def test_search_as_of_before_all_episodes_returns_empty() -> None:
    store = _make_store()
    _insert_ep(store, "ep1", _T1, "something")

    vec = np.zeros(384, dtype=np.float32)
    results = store.search_episodes_as_of(vec, k=5, as_of=_T0)
    assert results == []


def test_search_as_of_after_all_episodes_returns_all() -> None:
    store = _make_store()
    _insert_ep(store, "ep1", _T0, "first")
    _insert_ep(store, "ep2", _T1, "second")

    vec = np.zeros(384, dtype=np.float32)
    results = store.search_episodes_as_of(vec, k=5, as_of=_T3)
    ids = {ep.id for ep, _, _ in results}
    assert ids == {"ep1", "ep2"}


def test_search_as_of_respects_k_limit() -> None:
    store = _make_store()
    for i in range(5):
        _insert_ep(store, f"ep{i}", _T0, f"content {i}")

    vec = np.zeros(384, dtype=np.float32)
    results = store.search_episodes_as_of(vec, k=3, as_of=_T1)
    assert len(results) <= 3


def test_search_as_of_inclusive_boundary() -> None:
    """Episode at exactly as_of should be included."""
    store = _make_store()
    _insert_ep(store, "ep_exact", _T1, "at boundary")

    vec = np.zeros(384, dtype=np.float32)
    results = store.search_episodes_as_of(vec, k=5, as_of=_T1)
    ids = {ep.id for ep, _, _ in results}
    assert "ep_exact" in ids


# ------------------------------------------------------------------
# get_facts_as_of
# ------------------------------------------------------------------


def test_get_facts_as_of_returns_valid_fact() -> None:
    store = _make_store()
    _make_fact(store, "f1", "Ivan", "works_at", "Acme", valid_from=_T0)

    facts = store.get_facts_as_of("Ivan", _T1)
    assert len(facts) == 1
    assert facts[0].object == "Acme"


def test_get_facts_as_of_time_travel() -> None:
    """At T1 → Acme; after T1 when superseded → Globex."""
    store = _make_store()
    # Ivan worked at Acme from T0 to T1
    _make_fact(
        store, "f1", "Ivan", "works_at", "Acme", valid_from=_T0, valid_to=_T1, superseded_by="f2"
    )
    # Ivan works at Globex from T1 onwards
    _make_fact(store, "f2", "Ivan", "works_at", "Globex", valid_from=_T1)

    at_t0 = store.get_facts_as_of("Ivan", _T0 + timedelta(hours=1))
    assert len(at_t0) == 1
    assert at_t0[0].object == "Acme"

    at_t2 = store.get_facts_as_of("Ivan", _T2)
    assert len(at_t2) == 1
    assert at_t2[0].object == "Globex"


def test_get_facts_as_of_before_any_fact_returns_empty() -> None:
    store = _make_store()
    _make_fact(store, "f1", "Ivan", "works_at", "Acme", valid_from=_T1)

    facts = store.get_facts_as_of("Ivan", _T0)
    assert facts == []


def test_get_facts_as_of_excludes_closed_fact_after_valid_to() -> None:
    store = _make_store()
    _make_fact(
        store, "f1", "Ivan", "works_at", "Acme", valid_from=_T0, valid_to=_T1, superseded_by="f2"
    )

    # Query after valid_to — should not return the closed fact
    facts = store.get_facts_as_of("Ivan", _T2)
    assert all(f.id != "f1" for f in facts)


def test_get_facts_as_of_unknown_subject_returns_empty() -> None:
    store = _make_store()
    assert store.get_facts_as_of("nobody", _T1) == []


def test_get_facts_as_of_naive_datetime_at_boundary() -> None:
    """A naive (tz-less) as_of must compare correctly against aware-UTC storage.

    Regression: timestamps are compared as TEXT, and a naive datetime serialises
    without the '+00:00' offset that stored aware timestamps carry, which flipped
    the boundary comparison and silently dropped facts valid exactly at as_of.
    """
    store = _make_store()
    _make_fact(
        store, "f1", "Ivan", "works_at", "Acme", valid_from=_T0, valid_to=_T2, superseded_by="f2"
    )
    naive_t1 = _T1.replace(tzinfo=None)  # valid_from < naive_t1 < valid_to
    facts = store.get_facts_as_of("Ivan", naive_t1)
    assert len(facts) == 1
    assert facts[0].object == "Acme"


def test_search_as_of_naive_datetime_inclusive_boundary() -> None:
    """An episode at exactly a naive as_of is included (boundary correctness)."""
    store = _make_store()
    _insert_ep(store, "e1", _T1, content="boundary episode")
    vec = np.zeros(384, dtype=np.float32)
    results = store.search_episodes_as_of(vec, k=5, as_of=_T1.replace(tzinfo=None))
    assert any(ep.id == "e1" for ep, _, _ in results)


# ------------------------------------------------------------------
# timeline via public Engram API
# ------------------------------------------------------------------


def test_timeline_returns_all_facts_in_order() -> None:
    from engram.core import Engram

    mem = Engram()
    mem.assert_fact("Ivan", "works_at", "Acme", confidence=0.9)
    mem.assert_fact("Ivan", "lives_in", "Berlin", confidence=0.8)

    history = mem.timeline("Ivan")
    assert len(history) == 2
    subjects = {f.subject for f in history}
    assert subjects == {"Ivan"}
    predicates = {f.predicate for f in history}
    assert predicates == {"works_at", "lives_in"}


def test_timeline_includes_superseded_facts() -> None:
    from engram.core import Engram
    from engram.llm import StubLLMAdapter

    stub = StubLLMAdapter(
        [
            {"subject": "Ivan", "predicate": "works_at", "object": "Acme", "confidence": 0.9},
        ]
    )
    mem = Engram(llm=stub)
    mem.observe("Ivan works at Acme")
    mem.reflect()

    stub2 = StubLLMAdapter(
        [
            {"subject": "Ivan", "predicate": "works_at", "object": "Globex", "confidence": 0.9},
        ]
    )
    mem._llm = stub2
    mem.observe("Ivan now works at Globex")
    mem.reflect()

    history = mem.timeline("Ivan")
    objects = [f.object for f in history]
    assert "Acme" in objects
    assert "Globex" in objects


def test_timeline_unknown_entity_returns_empty() -> None:
    from engram.core import Engram

    mem = Engram()
    assert mem.timeline("nobody") == []


def test_timeline_as_of_returns_only_valid_facts() -> None:
    """timeline(entity, as_of=T) routes through get_facts_as_of."""
    from engram.core import Engram

    mem = Engram()
    # Two facts: Ivan@Acme until T1, then Ivan@Globex from T1.
    mem._store.insert_fact(
        Fact(
            id="ft-a",
            subject="Ivan",
            predicate="works_at",
            object="Acme",
            valid_from=_T0,
            valid_to=_T1,
            recorded_at=_T0,
            superseded_at=_T1,
            superseded_by="ft-b",
            confidence=0.9,
            derived_from=[],
        )
    )
    mem._store.insert_fact(
        Fact(
            id="ft-b",
            subject="Ivan",
            predicate="works_at",
            object="Globex",
            valid_from=_T1,
            valid_to=None,
            recorded_at=_T1,
            superseded_at=None,
            superseded_by=None,
            confidence=0.9,
            derived_from=[],
        )
    )

    at_t0 = mem.timeline("Ivan", as_of=_T0 + timedelta(hours=1))
    assert len(at_t0) == 1
    assert at_t0[0].object == "Acme"

    at_t2 = mem.timeline("Ivan", as_of=_T2)
    assert len(at_t2) == 1
    assert at_t2[0].object == "Globex"

    # Full timeline (no as_of) returns both regardless of supersession.
    assert {f.object for f in mem.timeline("Ivan")} == {"Acme", "Globex"}


# ------------------------------------------------------------------
# recall(as_of=T) via public Engram API
# ------------------------------------------------------------------


def test_recall_as_of_filters_by_time(tmp_path: pytest.TempPathFactory) -> None:
    from engram.core import Engram

    db = str(tmp_path / "mem.engram")  # type: ignore[operator]
    mem = Engram(path=db)

    # Manually insert episodes with controlled timestamps via store
    from engram.models import Episode

    ep_old = Episode(id="old", content="past event here", timestamp=_T0)
    ep_new = Episode(id="new", content="future event here", timestamp=_T2)
    vec = np.zeros(384, dtype=np.float32)
    mem._store.insert_episode(ep_old, vec)
    mem._store.insert_episode(ep_new, vec)

    results = mem.recall("event", k=5, as_of=_T1)
    ids = {r.episode.id for r in results}
    assert "old" in ids
    assert "new" not in ids


def test_recall_as_of_spreading_no_crash(tmp_path: pytest.TempPathFactory) -> None:
    from engram.core import Engram
    from engram.models import Episode

    db = str(tmp_path / "mem.engram")  # type: ignore[operator]
    mem = Engram(path=db)

    ep = Episode(id="ep1", content="some event", timestamp=_T0)
    vec = np.zeros(384, dtype=np.float32)
    mem._store.insert_episode(ep, vec)

    results = mem.recall("event", k=5, mode="spreading", as_of=_T1)
    assert isinstance(results, list)
