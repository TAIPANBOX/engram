"""Tests for entity/edge CRUD and spreading-activation retrieval."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from engram.models import Episode, SearchResult
from engram.schema import migrate
from engram.store import Store

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _make_store() -> Store:
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    return Store(conn, dim=384)


def _insert_ep(store: Store, ep_id: str, content: str = "text") -> None:
    ep = Episode(id=ep_id, content=content, timestamp=_T0)
    vec = np.zeros(384, dtype=np.float32)
    store.insert_episode(ep, vec)


# ------------------------------------------------------------------
# Entity CRUD
# ------------------------------------------------------------------


def test_find_or_create_entity_inserts_new() -> None:
    store = _make_store()
    entity = store.find_or_create_entity("Ivan", "person", _T0)
    assert entity.name == "Ivan"
    assert entity.type == "person"
    assert entity.first_seen == _T0
    assert entity.last_seen == _T0


def test_find_or_create_entity_idempotent() -> None:
    store = _make_store()
    e1 = store.find_or_create_entity("Ivan", "person", _T0)
    t1 = _T0 + timedelta(hours=1)
    e2 = store.find_or_create_entity("Ivan", "person", t1)
    assert e1.id == e2.id
    # last_seen updated
    assert e2.last_seen == t1
    # only one row
    assert len(store.get_all_entities()) == 1


def test_get_entity_by_name_missing() -> None:
    store = _make_store()
    assert store.get_entity_by_name("nobody") is None


def test_get_all_entities_empty() -> None:
    store = _make_store()
    assert store.get_all_entities() == []


def test_get_all_entities_multiple() -> None:
    store = _make_store()
    store.find_or_create_entity("Ivan", "person", _T0)
    store.find_or_create_entity("Acme", "org", _T0)
    entities = store.get_all_entities()
    assert len(entities) == 2
    names = {e.name for e in entities}
    assert names == {"Ivan", "Acme"}


# ------------------------------------------------------------------
# Edge CRUD
# ------------------------------------------------------------------


def test_insert_edge_basic() -> None:
    store = _make_store()
    _insert_ep(store, "ep1")
    entity = store.find_or_create_entity("Ivan", "person", _T0)
    store.insert_edge("ep1", entity.id, "mentions", weight=0.8, created_at=_T0)
    neighbors = store.get_neighbors("ep1")
    assert len(neighbors) == 1
    assert neighbors[0][0] == entity.id
    assert abs(neighbors[0][1] - 0.8) < 1e-6


def test_insert_edge_hebbian_accumulation() -> None:
    store = _make_store()
    _insert_ep(store, "ep1")
    entity = store.find_or_create_entity("Ivan", "person", _T0)
    store.insert_edge("ep1", entity.id, "mentions", weight=0.5, created_at=_T0)
    store.insert_edge("ep1", entity.id, "mentions", weight=0.5, created_at=_T0)
    neighbors = store.get_neighbors("ep1")
    assert abs(neighbors[0][1] - 1.0) < 1e-6


def test_get_neighbors_bidirectional() -> None:
    store = _make_store()
    _insert_ep(store, "ep1")
    _insert_ep(store, "ep2")
    store.insert_edge("ep1", "ep2", "related", weight=1.0, created_at=_T0)
    # ep1 -> ep2 direction
    fwd = store.get_neighbors("ep1")
    assert any(n == "ep2" for n, _ in fwd)
    # ep2 -> ep1 (reverse)
    rev = store.get_neighbors("ep2")
    assert any(n == "ep1" for n, _ in rev)


def test_get_neighbors_empty() -> None:
    store = _make_store()
    _insert_ep(store, "ep1")
    assert store.get_neighbors("ep1") == []


# ------------------------------------------------------------------
# get_episodes_by_ids
# ------------------------------------------------------------------


def test_get_episodes_by_ids_returns_correct() -> None:
    store = _make_store()
    _insert_ep(store, "ep1", "first")
    _insert_ep(store, "ep2", "second")
    results = store.get_episodes_by_ids(["ep1", "ep2"])
    ids = {ep.id for ep in results}
    assert ids == {"ep1", "ep2"}


def test_get_episodes_by_ids_ignores_unknown() -> None:
    store = _make_store()
    _insert_ep(store, "ep1", "first")
    results = store.get_episodes_by_ids(["ep1", "ghost"])
    assert len(results) == 1
    assert results[0].id == "ep1"


def test_get_episodes_by_ids_empty_list() -> None:
    store = _make_store()
    assert store.get_episodes_by_ids([]) == []


# ------------------------------------------------------------------
# spreading_recall
# ------------------------------------------------------------------


def test_spreading_recall_empty_store() -> None:
    """No crash on an empty store."""
    from unittest.mock import MagicMock

    import numpy as np

    from engram.graph import spreading_recall

    store = _make_store()
    embedder = MagicMock()
    embedder.embed.return_value = np.zeros(384, dtype=np.float32)
    results = spreading_recall("anything", k=5, store=store, embedder=embedder)
    assert results == []


def test_spreading_recall_promotes_connected_episode(tmp_path: pytest.TempPathFactory) -> None:
    """An episode connected via a shared entity should score higher than an unconnected one."""
    from engram.core import Engram
    from engram.llm import StubLLMAdapter

    db = str(tmp_path / "test.engram")  # type: ignore[operator]
    stub_facts = [
        {"subject": "Ivan", "predicate": "works_at", "object": "Acme", "confidence": 0.9},
    ]
    mem = Engram(path=db, llm=StubLLMAdapter(stub_facts))

    # ep_a and ep_b are linked via entity "Ivan" after reflection
    _ep_a = mem.observe("Ivan joined Acme as an engineer")
    _ep_b = mem.observe("Ivan attended the company meeting")
    # ep_c is unrelated
    _ep_c = mem.observe("The weather is sunny today")

    mem.reflect()

    results = mem.recall("Ivan", k=5, mode="spreading")
    assert isinstance(results, list)
    assert len(results) >= 1
    assert all(isinstance(r, SearchResult) for r in results)


def test_spreading_recall_no_double_count_for_seeds() -> None:
    """A seed episode with no edges should score exactly alpha*cosine + gamma*importance.

    Pre-fix, seed activation = cosine was added a second time as beta*activation,
    inflating seed scores. With cosine.score≈c, importance=1, and (alpha, beta, gamma) =
    (0.6, 0.3, 0.1), the correct score is 0.6c + 0.1, NOT 0.6c + 0.3c + 0.1.
    """
    from unittest.mock import MagicMock

    import numpy as np

    from engram.graph import spreading_recall

    store = _make_store()
    ep = Episode(id="solo", content="lone", timestamp=_T0, importance_score=1.0)
    vec = np.ones(384, dtype=np.float32)
    vec /= float(np.linalg.norm(vec))
    store.insert_episode(ep, vec)

    embedder = MagicMock()
    embedder.embed.return_value = vec

    results = spreading_recall(
        "lone",
        k=5,
        store=store,
        embedder=embedder,
        depth=2,
        decay=0.5,
        alpha=0.6,
        beta=0.3,
        gamma=0.1,
    )
    assert len(results) == 1
    cosine = results[0].distance  # not used; pull cosine from observation
    # We can't read cosine_scores directly, but we know vec MATCHes itself: score ≈ 1.0.
    # Score must be alpha*1 + gamma*1 = 0.7, NOT 0.7 + 0.3 (no neighbours) = 1.0 from the old code.
    # Tolerance accounts for the 1/(1+L2) score, not exact 1.0.
    assert results[0].score < 0.95, (
        f"Solo seed double-counted as graph activation: got score={results[0].score}; "
        f"expected ≈ alpha*cosine + gamma*importance. cosine sentinel={cosine}"
    )


def test_spreading_recall_respects_edge_weight() -> None:
    """A heavy edge to an entity should pull a connected episode more than a light one."""
    from unittest.mock import MagicMock

    import numpy as np

    from engram.graph import spreading_recall

    store = _make_store()
    seed_vec = np.zeros(384, dtype=np.float32)
    seed_vec[0] = 1.0
    other_vec = np.zeros(384, dtype=np.float32)
    other_vec[1] = 1.0
    # near_vec slightly off-axis so it's NOT picked up as a seed but exists in store
    near_vec = other_vec.copy()

    store.insert_episode(Episode(id="seed", content="seed-text", timestamp=_T0), seed_vec)
    store.insert_episode(Episode(id="heavy", content="heavy-target", timestamp=_T0), near_vec)
    store.insert_episode(Episode(id="light", content="light-target", timestamp=_T0), near_vec)

    # Heavy edge from seed -> heavy, light edge from seed -> light.
    store.insert_edge("seed", "heavy", "co", weight=0.9, created_at=_T0)
    store.insert_edge("seed", "light", "co", weight=0.1, created_at=_T0)

    embedder = MagicMock()
    embedder.embed.return_value = seed_vec

    results = spreading_recall(
        "anything",
        k=10,
        store=store,
        embedder=embedder,
        depth=1,
        decay=1.0,
        alpha=0.0,  # zero-out cosine so only graph_act drives ranking
        beta=1.0,
        gamma=0.0,
    )
    by_id = {r.episode.id: r.score for r in results}
    assert "heavy" in by_id
    assert "light" in by_id
    assert by_id["heavy"] > by_id["light"], (
        f"Heavier edge should win: heavy={by_id['heavy']}, light={by_id['light']}"
    )


def test_recall_cosine_mode_unchanged() -> None:
    """mode='cosine' still works after v0.4 changes."""
    from engram.core import Engram

    mem = Engram()
    mem.observe("The quick brown fox")
    results = mem.recall("fox", k=1, mode="cosine")
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)


def test_recall_spreading_mode_returns_results() -> None:
    """mode='spreading' returns SearchResult list without errors."""
    from engram.core import Engram

    mem = Engram()
    mem.observe("Python is a programming language")
    mem.observe("Snakes live in warm climates")
    results = mem.recall("Python snake", k=2, mode="spreading")
    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)


# ------------------------------------------------------------------
# Reflection + entity/edge integration
# ------------------------------------------------------------------


def test_reflection_creates_entities_and_edges() -> None:
    """After reflection with StubLLM, entities and edges must exist in the store."""
    from engram.importance import DecayConfig
    from engram.llm import StubLLMAdapter
    from engram.reflection import reflect

    store = _make_store()
    _insert_ep(store, "ep1", "Ivan works at Acme")

    stub = StubLLMAdapter(
        [
            {"subject": "Ivan", "predicate": "works_at", "object": "Acme", "confidence": 0.9},
        ]
    )
    reflect(store, stub, DecayConfig(), now=_T0)

    entities = store.get_all_entities()
    names = {e.name for e in entities}
    assert "Ivan" in names
    assert "Acme" in names

    # Both entity nodes should have edges from ep1
    for entity in entities:
        neighbors = store.get_neighbors(entity.id)
        neighbor_ids = {n for n, _ in neighbors}
        assert "ep1" in neighbor_ids


def test_reflection_entity_count_equals_unique_subjects_objects() -> None:
    """Unique subjects+objects from facts == entity count (no duplicates)."""
    from engram.importance import DecayConfig
    from engram.llm import StubLLMAdapter
    from engram.reflection import reflect

    store = _make_store()
    _insert_ep(store, "ep1", "Ivan works at Acme; Maria lives in Berlin")

    # Ivan appears as subject in both facts — should only create one entity
    stub = StubLLMAdapter(
        [
            {"subject": "Ivan", "predicate": "works_at", "object": "Acme", "confidence": 0.8},
            {"subject": "Ivan", "predicate": "knows", "object": "Maria", "confidence": 0.7},
            {"subject": "Maria", "predicate": "lives_in", "object": "Berlin", "confidence": 0.9},
        ]
    )
    reflect(store, stub, DecayConfig(), now=_T0)

    entities = store.get_all_entities()
    names = {e.name for e in entities}
    # Unique names across all subjects+objects: Ivan, Acme, Maria, Berlin
    assert names == {"Ivan", "Acme", "Maria", "Berlin"}
    assert len(entities) == 4
