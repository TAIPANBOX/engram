"""Tests for the reflection loop and related public API."""

import threading

import pytest

from engram import Engram, ReflectionRun, StubLLMAdapter


@pytest.fixture()
def mem() -> Engram:
    with Engram(path=":memory:") as m:
        yield m


@pytest.fixture()
def mem_with_llm() -> Engram:
    stub = StubLLMAdapter(
        facts=[
            {"subject": "Ivan", "predicate": "works_at", "object": "Globex", "confidence": 0.9},
        ]
    )
    with Engram(path=":memory:", llm=stub) as m:
        yield m


# ------------------------------------------------------------------
# reflect() without LLM
# ------------------------------------------------------------------


def test_reflect_no_llm_returns_run(mem: Engram) -> None:
    mem.observe("Something happened today")
    run = mem.reflect()
    assert isinstance(run, ReflectionRun)
    assert run.finished_at is not None


def test_reflect_no_llm_no_facts_extracted(mem: Engram) -> None:
    mem.observe("Something happened today")
    run = mem.reflect()
    assert run.facts_extracted == 0


def test_reflect_no_llm_empty_store(mem: Engram) -> None:
    run = mem.reflect()
    assert run.episodes_processed == 0


# ------------------------------------------------------------------
# reflect() with StubLLM
# ------------------------------------------------------------------


def test_reflect_with_llm_extracts_facts(mem_with_llm: Engram) -> None:
    mem_with_llm.observe("Ivan mentioned he now works at Globex")
    run = mem_with_llm.reflect()
    assert run.facts_extracted == 1


def test_reflect_inserts_fact_into_store(mem_with_llm: Engram) -> None:
    mem_with_llm.observe("Ivan mentioned he now works at Globex")
    mem_with_llm.reflect()
    facts = mem_with_llm._store.get_all_facts("Ivan")
    assert len(facts) == 1
    assert facts[0].object == "Globex"


def test_reflect_second_run_processes_only_new_episodes() -> None:
    stub = StubLLMAdapter(
        facts=[
            {"subject": "Ivan", "predicate": "works_at", "object": "Acme", "confidence": 0.9},
        ]
    )
    with Engram(path=":memory:", llm=stub) as mem:
        mem.observe("Ivan works at Acme")
        run1 = mem.reflect()
        assert run1.episodes_processed == 1

        mem.observe("New event after first reflection")
        run2 = mem.reflect()
        assert run2.episodes_processed == 1  # only the new episode


def test_reflect_contradiction_detection() -> None:
    """When the LLM extracts a new (s,p) that already exists, close the old fact."""
    stub_acme = StubLLMAdapter(
        facts=[
            {"subject": "Ivan", "predicate": "works_at", "object": "Acme", "confidence": 0.9},
        ]
    )
    stub_globex = StubLLMAdapter(
        facts=[
            {"subject": "Ivan", "predicate": "works_at", "object": "Globex", "confidence": 0.95},
        ]
    )
    with Engram(path=":memory:", llm=stub_acme) as mem:
        mem.observe("Ivan works at Acme")
        mem.reflect()

        # Switch LLM stub to return conflicting fact
        mem._llm = stub_globex
        mem.observe("Ivan is now at Globex")
        run = mem.reflect()

        assert run.contradictions_resolved == 1
        active = mem._store.get_active_facts("Ivan", "works_at")
        assert len(active) == 1
        assert active[0].object == "Globex"
        # Old fact should be closed
        all_facts = mem._store.get_all_facts("Ivan")
        closed = [f for f in all_facts if f.valid_to is not None]
        assert len(closed) == 1
        assert closed[0].object == "Acme"


# ------------------------------------------------------------------
# reflect_async()
# ------------------------------------------------------------------


def test_reflect_async_returns_thread(mem: Engram) -> None:
    mem.observe("Background event")
    t = mem.reflect_async()
    assert isinstance(t, threading.Thread)
    t.join(timeout=30)
    assert not t.is_alive()


def test_reflect_async_completes_successfully(mem: Engram) -> None:
    for i in range(5):
        mem.observe(f"Event {i}")
    t = mem.reflect_async()
    t.join(timeout=60)
    assert mem._store.get_last_reflection() is not None


# ------------------------------------------------------------------
# assert_fact() / why() / contradictions()
# ------------------------------------------------------------------


def test_assert_fact_returns_id(mem: Engram) -> None:
    fid = mem.assert_fact("Ivan", "lives_in", "Kyiv")
    assert isinstance(fid, str) and len(fid) > 0


def test_assert_fact_stored_correctly(mem: Engram) -> None:
    fid = mem.assert_fact("Maria", "role", "CTO", confidence=0.8, source="conversation-123")
    fact = mem._store.get_fact(fid)
    assert fact is not None
    assert fact.subject == "Maria"
    assert fact.predicate == "role"
    assert fact.object == "CTO"
    assert fact.confidence == pytest.approx(0.8)
    assert "conversation-123" in fact.derived_from


def test_why_returns_provenance(mem_with_llm: Engram) -> None:
    mem_with_llm.observe("Ivan now works at Globex")
    mem_with_llm.reflect()
    facts = mem_with_llm._store.get_all_facts("Ivan")
    assert facts
    prov = mem_with_llm.why(facts[0].id)
    assert "fact" in prov
    assert "extracted_from" in prov
    assert "confidence" in prov
    assert "model" in prov
    assert prov["model"] == "stub"


def test_why_manually_asserted_fact(mem: Engram) -> None:
    fid = mem.assert_fact("Lena", "department", "Engineering")
    prov = mem.why(fid)
    assert prov["extracted_by"] is None
    assert prov["model"] is None
    assert prov["fact"] == "Lena department Engineering"


def test_why_raises_for_missing_fact(mem: Engram) -> None:
    with pytest.raises(KeyError):
        mem.why("nonexistent-id")


def test_contradictions_empty_when_no_conflicts(mem: Engram) -> None:
    mem.assert_fact("Ivan", "works_at", "Globex")
    mem.assert_fact("Ivan", "lives_in", "Berlin")
    assert mem.contradictions() == []


def test_contradictions_detects_conflict(mem: Engram) -> None:
    mem.assert_fact("Ivan", "works_at", "Acme")
    mem.assert_fact("Ivan", "works_at", "Globex")  # same (s,p), different object
    pairs = mem.contradictions()
    assert len(pairs) == 1
    subjects = {pairs[0][0].object, pairs[0][1].object}
    assert subjects == {"Acme", "Globex"}


def test_contradictions_resolved_after_reflect() -> None:
    stub = StubLLMAdapter(
        facts=[
            {"subject": "Ivan", "predicate": "works_at", "object": "Globex", "confidence": 0.9},
        ]
    )
    with Engram(path=":memory:", llm=stub) as mem:
        mem.assert_fact("Ivan", "works_at", "Acme")
        mem.observe("Ivan moved to Globex")
        mem.reflect()
        # The reflected fact closes the manually asserted one
        # (reflection closes any pre-existing active facts with same s,p)
        pairs = mem.contradictions()
        assert len(pairs) == 0


def test_reflection_run_model_recorded(mem_with_llm: Engram) -> None:
    mem_with_llm.observe("Some event")
    run = mem_with_llm.reflect()
    assert run.model_used == "stub"
    stored = mem_with_llm._store.get_last_reflection()
    assert stored is not None
    assert stored.model_used == "stub"
