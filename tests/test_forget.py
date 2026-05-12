"""Tests for forget() and forget_entity() — v1.1 GDPR erasure."""

from __future__ import annotations

import pytest

from engram import Engram, ForgetResult, StubLLMAdapter


@pytest.fixture()
def mem() -> Engram:
    with Engram(path=":memory:") as m:
        yield m


# ------------------------------------------------------------------
# forget() — single episode
# ------------------------------------------------------------------


def test_forget_removes_episode_from_recall(mem: Engram) -> None:
    ep_id = mem.observe("Ivan joined Globex last week", actors=["Ivan"])
    mem.forget(ep_id)
    results = mem.recall("Ivan Globex", k=5)
    ids = [r.episode.id for r in results]
    assert ep_id not in ids


def test_forget_raises_for_unknown_id(mem: Engram) -> None:
    with pytest.raises(KeyError):
        mem.forget("nonexistent-uuid")


def test_forget_raises_if_already_forgotten(mem: Engram) -> None:
    ep_id = mem.observe("One-off event")
    mem.forget(ep_id)
    with pytest.raises(KeyError):
        mem.forget(ep_id)


def test_forget_removes_from_vec_index(mem: Engram) -> None:
    mem.observe("Background noise A")
    mem.observe("Background noise B")
    target_id = mem.observe("Target episode to erase")
    before_vec = mem._store.vec_count()
    mem.forget(target_id)
    assert mem._store.vec_count() == before_vec - 1


def test_forget_episode_count_decreases(mem: Engram) -> None:
    ep_id = mem.observe("Temporary event")
    before = mem._store.episode_count()
    mem.forget(ep_id)
    assert mem._store.episode_count() == before - 1


def test_forget_with_reason_does_not_raise(mem: Engram) -> None:
    ep_id = mem.observe("Sensitive data")
    mem.forget(ep_id, reason="user requested deletion")  # must not raise


def test_forget_other_episodes_unaffected(mem: Engram) -> None:
    keep_id = mem.observe("Keep this one")
    drop_id = mem.observe("Delete this one")
    mem.forget(drop_id)
    results = mem.recall("Keep this one", k=5)
    assert any(r.episode.id == keep_id for r in results)


# ------------------------------------------------------------------
# forget_entity() — GDPR erasure
# ------------------------------------------------------------------


def test_forget_entity_returns_forget_result(mem: Engram) -> None:
    mem.observe("Ivan joined Globex", actors=["Ivan"])
    result = mem.forget_entity("Ivan")
    assert isinstance(result, ForgetResult)
    assert result.entity == "Ivan"


def test_forget_entity_deletes_actor_episodes(mem: Engram) -> None:
    mem.observe("Ivan joined Globex", actors=["Ivan"])
    mem.observe("Ivan presented the roadmap", actors=["Ivan"])
    mem.observe("Maria reviewed the code", actors=["Maria"])

    result = mem.forget_entity("Ivan")

    assert result.episodes_deleted == 2
    # Ivan's episodes must not appear in recall
    for r in mem.recall("Ivan", k=10):
        assert "Ivan" not in r.episode.actors


def test_forget_entity_leaves_other_actors_intact(mem: Engram) -> None:
    ivan_id = mem.observe("Ivan joined Globex", actors=["Ivan"])
    maria_id = mem.observe("Maria reviewed the code", actors=["Maria"])

    mem.forget_entity("Ivan")

    # Maria's episode survives
    remaining = [r.episode.id for r in mem.recall("code review", k=10)]
    assert maria_id in remaining
    assert ivan_id not in remaining


def test_forget_entity_deletes_facts(mem: Engram) -> None:
    stub = StubLLMAdapter(
        facts=[
            {"subject": "Ivan", "predicate": "works_at", "object": "Globex", "confidence": 0.9},
            {"subject": "Ivan", "predicate": "lives_in", "object": "Berlin", "confidence": 0.8},
        ]
    )
    with Engram(path=":memory:", llm=stub) as m:
        m.observe("Ivan works at Globex and lives in Berlin", actors=["Ivan"])
        m.reflect()

        result = m.forget_entity("Ivan")

        assert result.facts_deleted == 2
        assert m._store.get_all_facts("Ivan") == []


def test_forget_entity_deletes_object_facts(mem: Engram) -> None:
    """Facts where Ivan is the object (not subject) must also be erased."""
    mem.assert_fact("Alice", "reports_to", "Ivan")
    result = mem.forget_entity("Ivan")
    assert result.facts_deleted == 1


def test_forget_entity_deletes_entity_record(mem: Engram) -> None:
    mem.observe("Ivan joined Globex", actors=["Ivan"])
    mem.reflect()  # no LLM → no facts, but entity may exist from spreading
    mem.forget_entity("Ivan")
    assert mem._store.get_entity_by_name("Ivan") is None


def test_forget_entity_removes_from_vec_index(mem: Engram) -> None:
    mem.observe("Ivan joined Globex", actors=["Ivan"])
    mem.observe("Ivan presented Q3 roadmap", actors=["Ivan"])
    before_vec = mem._store.vec_count()
    result = mem.forget_entity("Ivan")
    assert mem._store.vec_count() == before_vec - result.episodes_deleted


def test_forget_entity_unknown_entity_returns_zero_counts(mem: Engram) -> None:
    result = mem.forget_entity("Nobody")
    assert result.episodes_deleted == 0
    assert result.facts_deleted == 0


def test_forget_entity_assert_fact_erased(mem: Engram) -> None:
    fact_id = mem.assert_fact("Ivan", "works_at", "Globex")
    mem.forget_entity("Ivan")
    with pytest.raises(KeyError):
        mem.why(fact_id)


# ------------------------------------------------------------------
# prune_episodes vec_index fix (regression test)
# ------------------------------------------------------------------


def test_prune_removes_from_vec_index(mem: Engram) -> None:
    """prune_episodes() must delete from vec_episodes, not just episodes."""
    from engram.importance import DecayConfig

    cfg = DecayConfig(lambda_=100.0, threshold=0.9)  # aggressive decay
    mem.observe("Trivial event", salience=0.01)

    before_vec = mem._store.vec_count()
    from engram.decay import run_decay

    run_decay(mem._store, cfg)
    pruned = mem._store.prune_episodes(cfg.threshold)

    assert pruned > 0
    assert mem._store.vec_count() == before_vec - pruned
    assert mem._store.episode_count() == mem._store.vec_count()
