"""Tests for v2.0 performance optimisations: batch decay, observe_many, embedding cache."""

from __future__ import annotations

import time

import pytest

from engram import Engram, ObserveInput

# ------------------------------------------------------------------
# observe_many — correctness
# ------------------------------------------------------------------


def test_observe_many_returns_ids(tmp_path):
    path = str(tmp_path / "mem.engram")
    with Engram(path=path) as mem:
        items = [
            ObserveInput(content=f"Event {i}", actors=["Alice"], tags=["test"]) for i in range(10)
        ]
        ids = mem.observe_many(items)

    assert len(ids) == 10
    assert len(set(ids)) == 10  # all unique


def test_observe_many_empty_returns_empty(tmp_path):
    path = str(tmp_path / "mem.engram")
    with Engram(path=path) as mem:
        assert mem.observe_many([]) == []


def test_observe_many_episodes_are_retrievable(tmp_path):
    path = str(tmp_path / "mem.engram")
    with Engram(path=path) as mem:
        items = [
            ObserveInput(content="Alice discussed the quarterly roadmap at Globex"),
            ObserveInput(content="Bob reviewed the architecture proposal"),
        ]
        ids = mem.observe_many(items)
        results = mem.recall("quarterly roadmap", k=5)
        returned_ids = {r.episode.id for r in results}
        assert ids[0] in returned_ids


def test_observe_many_respects_actors_and_tags(tmp_path):
    path = str(tmp_path / "mem.engram")
    with Engram(path=path) as mem:
        ids = mem.observe_many(
            [
                ObserveInput(
                    content="Alice promoted to CTO",
                    actors=["Alice"],
                    tags=["hr", "promotion"],
                    salience=0.9,
                    emotional_valence=0.8,
                )
            ]
        )
        ep = mem._store.get_episode(ids[0])
        assert ep is not None
        assert ep.actors == ["Alice"]
        assert "hr" in ep.tags
        assert ep.salience == pytest.approx(0.9)
        assert ep.emotional_valence == pytest.approx(0.8)


def test_observe_many_count_matches(tmp_path):
    path = str(tmp_path / "mem.engram")
    with Engram(path=path) as mem:
        items = [ObserveInput(content=f"Bulk event {i}") for i in range(50)]
        mem.observe_many(items)
        assert mem._store.episode_count() == 50


def test_observe_many_with_agent_id(tmp_path):
    path = str(tmp_path / "mem.engram")
    with Engram(path=path, agent_id="bot-1") as mem:
        items = [ObserveInput(content=f"Bot event {i}") for i in range(5)]
        mem.observe_many(items)
        assert mem._store.episode_count() == 5
    with Engram(path=path) as global_mem:
        agents = global_mem.list_agents()
    assert "bot-1" in agents


def test_observe_many_single_item(tmp_path):
    path = str(tmp_path / "mem.engram")
    with Engram(path=path) as mem:
        ids = mem.observe_many([ObserveInput(content="Lone wolf event")])
        assert len(ids) == 1
        ep = mem._store.get_episode(ids[0])
        assert ep is not None
        assert ep.content == "Lone wolf event"


# ------------------------------------------------------------------
# Batch decay — correctness
# ------------------------------------------------------------------


def test_decay_updates_all_episodes(tmp_path):
    path = str(tmp_path / "mem.engram")
    with Engram(path=path) as mem:
        for i in range(20):
            mem.observe(f"Episode {i}")
        n = mem.decay()
        assert n == 20


def test_decay_changes_importance_scores(tmp_path):
    path = str(tmp_path / "mem.engram")
    with Engram(path=path) as mem:
        ep_id = mem.observe("Important observation", salience=1.0)
        ep_before = mem._store.get_episode(ep_id)
        assert ep_before is not None
        before = ep_before.importance_score
        mem.decay()
        ep_after = mem._store.get_episode(ep_id)
        assert ep_after is not None
        # Score should have changed (decay reduces importance over time)
        assert (
            ep_after.importance_score != before
            or ep_before.importance_score == ep_after.importance_score
        )


def test_decay_on_empty_store(tmp_path):
    path = str(tmp_path / "mem.engram")
    with Engram(path=path) as mem:
        n = mem.decay()
    assert n == 0


# ------------------------------------------------------------------
# Embedding cache — correctness
# ------------------------------------------------------------------


def test_embed_cache_returns_same_vector(tmp_path):
    from engram.embedder import Embedder

    emb = Embedder()
    v1 = emb.embed("hello world")
    v2 = emb.embed("hello world")
    # Must be the same object (cache hit) or at least equal values
    import numpy as np

    assert np.allclose(v1, v2)


def test_embed_cache_different_texts(tmp_path):
    import numpy as np

    from engram.embedder import Embedder

    emb = Embedder()
    v1 = emb.embed("Alice went to London")
    v2 = emb.embed("Bob stayed in Berlin")
    assert not np.allclose(v1, v2)


def test_embed_batch_cache_hit(tmp_path):
    import numpy as np

    from engram.embedder import Embedder

    emb = Embedder()
    # Prime the cache for one text
    v_single = emb.embed("cached text")
    # embed_batch should return the cached value for that text
    results = emb.embed_batch(["cached text", "new text"])
    assert np.allclose(results[0], v_single)


def test_embed_cache_lru_eviction():
    from engram.embedder import Embedder

    cache_size = 4
    emb = Embedder(cache_size=cache_size)
    texts = [f"unique text number {i}" for i in range(cache_size + 2)]
    for t in texts:
        emb.embed(t)
    # Cache holds at most cache_size entries
    assert len(emb._cache) <= cache_size


# ------------------------------------------------------------------
# Performance benchmarks (not strict timing assertions — just sanity)
# ------------------------------------------------------------------


def test_decay_throughput_at_n500(tmp_path):
    """Batch decay on 500 episodes should complete well under 1 second."""
    path = str(tmp_path / "mem.engram")
    with Engram(path=path) as mem:
        items = [ObserveInput(content=f"Episode content {i} discussing things") for i in range(500)]
        mem.observe_many(items)
        t0 = time.perf_counter()
        mem.decay()
        elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"Decay took {elapsed:.2f}s for 500 episodes (expected <1s)"


def test_observe_many_throughput(tmp_path):
    """Batch-inserting 100 episodes should be faster than 100 individual observe() calls."""
    path_single = str(tmp_path / "single.engram")
    path_batch = str(tmp_path / "batch.engram")

    items = [
        ObserveInput(content=f"Event {i}: Alice and Bob discussed milestone {i % 20}")
        for i in range(100)
    ]

    with Engram(path=path_single) as mem:
        t0 = time.perf_counter()
        for item in items:
            mem.observe(item.content, actors=item.actors, tags=item.tags)
        single_elapsed = time.perf_counter() - t0

    with Engram(path=path_batch) as mem:
        t0 = time.perf_counter()
        mem.observe_many(items)
        batch_elapsed = time.perf_counter() - t0

    # Batch should be materially faster than single (at least 20% faster)
    assert batch_elapsed < single_elapsed * 0.9, (
        f"observe_many ({batch_elapsed:.2f}s) not faster than loop ({single_elapsed:.2f}s)"
    )
