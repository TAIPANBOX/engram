"""Tests for engram.events and its wiring into Engram's write/reflect paths.

Schema validation uses a vendored copy of the Agent Passport
``agent-event.schema.json`` (see ``tests/fixtures/agent-event.schema.json``,
copied from ``TAIPANBOX/agent-passport`` -- SPEC.md §6). Vendored rather than
fetched at test time: CI checks out only this repo, and validating a wire
contract should never depend on a live network call.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from pathlib import Path

import jsonschema
import pytest

from engram import Engram, ObserveInput, StubLLMAdapter
from engram.events import EventLog, canonicalize, chain_hash, resolve_events_path

_SCHEMA_PATH = Path(__file__).parent / "fixtures" / "agent-event.schema.json"
_SCHEMA = json.loads(_SCHEMA_PATH.read_text())

_AGENT_ID = "agent://acme-bank.example/support/tier1-bot"


def _read_ndjson(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate(event: dict[str, object]) -> None:
    jsonschema.validate(instance=event, schema=_SCHEMA)


def _assert_chain(events: list[dict[str, object]]) -> None:
    """Every event validates, and the prev_hash chain links head to tail."""
    assert events, "no events to check"
    assert "prev_hash" not in events[0]
    for event in events:
        _validate(event)
    for previous, current in itertools.pairwise(events):
        assert current["prev_hash"] == chain_hash(previous)


# ------------------------------------------------------------------
# resolve_events_path
# ------------------------------------------------------------------


def test_resolve_events_path_none_when_nothing_set(monkeypatch) -> None:
    monkeypatch.delenv("ENGRAM_EVENTS_PATH", raising=False)
    assert resolve_events_path(None) is None


def test_resolve_events_path_explicit_wins(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ENGRAM_EVENTS_PATH", str(tmp_path / "env.ndjson"))
    explicit = tmp_path / "explicit.ndjson"
    assert resolve_events_path(explicit) == explicit


def test_resolve_events_path_env_fallback(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / "env.ndjson"
    monkeypatch.setenv("ENGRAM_EVENTS_PATH", str(env_path))
    assert resolve_events_path(None) == env_path


# ------------------------------------------------------------------
# Off by default
# ------------------------------------------------------------------


def test_off_by_default_no_file_created(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ENGRAM_EVENTS_PATH", raising=False)
    db_path = tmp_path / "store.engram"
    with Engram(path=str(db_path), agent_id=_AGENT_ID) as mem:
        mem.observe("Nothing should be logged")
        mem.assert_fact("Ivan", "works_at", "Globex")
        assert mem._events is None
    # No stray *.ndjson file anywhere near the store.
    assert list(tmp_path.glob("*.ndjson")) == []


def test_off_by_default_when_agent_id_missing_even_with_path(tmp_path) -> None:
    """events_path alone does not fabricate an agent_id-less event."""
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", events_path=events_path) as mem:
        mem.observe("no agent id on this instance")
    assert not events_path.exists() or _read_ndjson(events_path) == []
    assert mem._events is not None
    assert mem._events.skipped_empty_agent_id == 1


# ------------------------------------------------------------------
# Skip on empty agent_id
# ------------------------------------------------------------------


def test_emit_skips_and_counts_when_agent_id_empty(tmp_path) -> None:
    log = EventLog(tmp_path / "events.ndjson")
    log.emit("memory_written", None, {"memory_id": "x", "kind": "episodic"})
    log.emit("memory_written", "", {"memory_id": "x", "kind": "episodic"})
    assert log.skipped_empty_agent_id == 2
    assert not log.path.exists()


# ------------------------------------------------------------------
# Fail-open
# ------------------------------------------------------------------


def test_emit_fails_open_on_unwritable_path(tmp_path, caplog) -> None:
    # Point the events file at a path whose parent directory does not
    # exist, so every append raises OSError (FileNotFoundError).
    bad_path = tmp_path / "nonexistent-dir" / "events.ndjson"
    log = EventLog(bad_path)
    with caplog.at_level(logging.WARNING, logger="engram.events"):
        log.emit("memory_written", _AGENT_ID, {"memory_id": "x", "kind": "episodic"})
    assert any("engram.events" in r.name for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_engram_operation_succeeds_despite_unwritable_events_path(tmp_path, caplog) -> None:
    bad_events_path = tmp_path / "missing-parent" / "events.ndjson"
    with Engram(path=":memory:", agent_id=_AGENT_ID, events_path=bad_events_path) as mem:
        with caplog.at_level(logging.WARNING, logger="engram.events"):
            episode_id = mem.observe("write must succeed even though events can't")
        assert isinstance(episode_id, str) and episode_id
        assert mem._store.get_episode(episode_id) is not None
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# ------------------------------------------------------------------
# Golden-line schema validation for each emission site
# ------------------------------------------------------------------


def test_observe_emits_valid_memory_written_episodic(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=_AGENT_ID, events_path=events_path) as mem:
        episode_id = mem.observe("Ivan mentioned he works at Globex")

    events = _read_ndjson(events_path)
    assert len(events) == 1
    event = events[0]
    _validate(event)
    assert event["type"] == "memory_written"
    assert event["severity"] == "info"
    assert event["source"] == "engram"
    assert event["schema"] == "taipanbox.dev/agent-event/v0.1"
    assert event["agent_id"] == _AGENT_ID
    assert event["data"] == {"memory_id": episode_id, "kind": "episodic"}
    assert "prev_hash" not in event


def test_assert_fact_emits_valid_memory_written_semantic(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=_AGENT_ID, events_path=events_path) as mem:
        fact_id = mem.assert_fact("Ivan", "works_at", "Globex")

    events = _read_ndjson(events_path)
    assert len(events) == 1
    event = events[0]
    _validate(event)
    assert event["type"] == "memory_written"
    assert event["data"] == {"memory_id": fact_id, "kind": "semantic"}


def test_forget_emits_valid_memory_forgotten_episodic(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=_AGENT_ID, events_path=events_path) as mem:
        episode_id = mem.observe("to be forgotten")
        mem.forget(episode_id)

    events = _read_ndjson(events_path)
    assert len(events) == 2
    forgotten = events[1]
    _validate(forgotten)
    assert forgotten["type"] == "memory_forgotten"
    assert forgotten["data"] == {"memory_id": episode_id, "kind": "episodic"}


def test_forget_missing_episode_emits_nothing(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    with (
        Engram(path=":memory:", agent_id=_AGENT_ID, events_path=events_path) as mem,
        pytest.raises(KeyError),
    ):
        mem.forget("does-not-exist")

    assert not events_path.exists()


def test_forget_fact_emits_valid_memory_forgotten_semantic(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=_AGENT_ID, events_path=events_path) as mem:
        fact_id = mem.assert_fact("Ivan", "works_at", "Globex")
        mem.forget_fact(fact_id)

    events = _read_ndjson(events_path)
    assert len(events) == 2
    forgotten = events[1]
    _validate(forgotten)
    assert forgotten["type"] == "memory_forgotten"
    assert forgotten["data"] == {"memory_id": fact_id, "kind": "semantic"}


def test_reflect_emits_valid_reflection_run_event(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=_AGENT_ID, events_path=events_path) as mem:
        mem.observe("Something happened today")
        run = mem.reflect()

    events = _read_ndjson(events_path)
    reflection_events = [e for e in events if e["type"] == "reflection_run"]
    assert len(reflection_events) == 1
    event = reflection_events[0]
    _validate(event)
    assert event["severity"] == "info"
    assert event["run_id"] == run.id
    assert event["data"]["episodes_processed"] == run.episodes_processed
    assert event["data"]["facts_extracted"] == run.facts_extracted


def test_reflect_emits_valid_contradiction_found_event(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    stub = StubLLMAdapter(
        facts=[{"subject": "Ivan", "predicate": "works_at", "object": "Initech", "confidence": 0.9}]
    )
    with Engram(path=":memory:", agent_id=_AGENT_ID, llm=stub, events_path=events_path) as mem:
        old_fact_id = mem.assert_fact("Ivan", "works_at", "Globex")
        mem.observe("Ivan now works at Initech")
        run = mem.reflect()

    events = _read_ndjson(events_path)
    contradiction_events = [e for e in events if e["type"] == "contradiction_found"]
    assert len(contradiction_events) == 1
    event = contradiction_events[0]
    _validate(event)
    assert event["severity"] == "medium"
    assert event["data"]["conflicting_memory_id"] == old_fact_id
    assert event["data"]["memory_id"] != old_fact_id
    assert event["run_id"] == run.id
    assert run.contradictions_resolved == 1


def test_same_object_reextraction_emits_no_contradiction_event(tmp_path) -> None:
    """Superseding an identical (s, p, o) is agreement: no event, no count."""
    events_path = tmp_path / "events.ndjson"
    stub = StubLLMAdapter(
        facts=[{"subject": "Ivan", "predicate": "works_at", "object": "Globex", "confidence": 0.9}]
    )
    with Engram(path=":memory:", agent_id=_AGENT_ID, llm=stub, events_path=events_path) as mem:
        mem.assert_fact("Ivan", "works_at", "Globex")  # same object the stub re-extracts
        mem.observe("Ivan still works at Globex")
        run = mem.reflect()

    events = _read_ndjson(events_path)
    assert all(e["type"] != "contradiction_found" for e in events)
    assert run.contradictions_resolved == 0


def test_mixed_agreement_and_conflict_emits_one_event_for_the_conflict(tmp_path) -> None:
    """One agreeing and one conflicting older fact: exactly one event, for the conflict."""
    events_path = tmp_path / "events.ndjson"
    stub = StubLLMAdapter(
        facts=[{"subject": "Ivan", "predicate": "works_at", "object": "Globex", "confidence": 0.9}]
    )
    with Engram(path=":memory:", agent_id=_AGENT_ID, llm=stub, events_path=events_path) as mem:
        mem.assert_fact("Ivan", "works_at", "Globex")  # agreement: superseded silently
        acme_id = mem.assert_fact("Ivan", "works_at", "Acme")  # conflict: counted + emitted
        mem.observe("Ivan confirmed he works at Globex")
        run = mem.reflect()

    events = _read_ndjson(events_path)
    contradiction_events = [e for e in events if e["type"] == "contradiction_found"]
    assert len(contradiction_events) == 1
    _validate(contradiction_events[0])
    assert contradiction_events[0]["data"]["conflicting_memory_id"] == acme_id
    assert run.contradictions_resolved == 1


def test_no_llm_reflect_emits_no_contradiction_event(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=_AGENT_ID, events_path=events_path) as mem:
        mem.observe("Something happened today")
        mem.reflect()

    events = _read_ndjson(events_path)
    assert all(e["type"] != "contradiction_found" for e in events)


# ------------------------------------------------------------------
# Bulk and cascade paths: the ones a governance consumer most needs and the
# ones that emitted nothing at all before.
# ------------------------------------------------------------------


def test_observe_many_emits_one_memory_written_per_episode(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=_AGENT_ID, events_path=events_path) as mem:
        ids = mem.observe_many(
            [
                ObserveInput(content="first turn"),
                ObserveInput(content="second turn"),
                ObserveInput(content="third turn"),
            ]
        )

    events = _read_ndjson(events_path)
    assert len(events) == 3
    _assert_chain(events)
    assert [e["type"] for e in events] == ["memory_written"] * 3
    assert [e["data"] for e in events] == [
        {"memory_id": ep_id, "kind": "episodic"} for ep_id in ids
    ]


def test_observe_many_continues_the_chain_from_earlier_events(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=_AGENT_ID, events_path=events_path) as mem:
        mem.observe("a single write first")
        mem.observe_many([ObserveInput(content="then a batch"), ObserveInput(content="of two")])
        mem.observe("and a single write after")

    events = _read_ndjson(events_path)
    assert len(events) == 4
    _assert_chain(events)


def test_observe_many_of_nothing_emits_nothing(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=_AGENT_ID, events_path=events_path) as mem:
        assert mem.observe_many([]) == []
    assert not events_path.exists()


def test_forget_entity_emits_memory_forgotten_for_every_erased_memory(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=_AGENT_ID, events_path=events_path) as mem:
        ep_one = mem.observe("Ivan joined Globex", actors=["Ivan"])
        ep_two = mem.observe("Ivan moved to Initech", actors=["Ivan"])
        mem.observe("nothing to do with the erasure", actors=["Alice"])
        fact_id = mem.assert_fact("Ivan", "works_at", "Globex")
        result = mem.forget_entity("Ivan")

    events = _read_ndjson(events_path)
    _assert_chain(events)
    forgotten = [e for e in events if e["type"] == "memory_forgotten"]
    assert len(forgotten) == result.episodes_deleted + result.facts_deleted
    episodic = [e["data"]["memory_id"] for e in forgotten if e["data"]["kind"] == "episodic"]
    semantic = [e["data"]["memory_id"] for e in forgotten if e["data"]["kind"] == "semantic"]
    assert set(episodic) == {ep_one, ep_two}
    assert semantic == [fact_id]
    # Every erasure names the subject request that caused it.
    assert all(e["data"]["entity"] == "Ivan" for e in forgotten)


def test_forget_entity_with_nothing_to_erase_emits_nothing(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=_AGENT_ID, events_path=events_path) as mem:
        mem.observe("Alice signed off", actors=["Alice"])
        result = mem.forget_entity("Ivan")

    assert result.episodes_deleted == 0
    assert result.facts_deleted == 0
    events = _read_ndjson(events_path)
    assert all(e["type"] != "memory_forgotten" for e in events)


def test_compress_emits_memory_forgotten_for_every_source_it_deletes(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    stub = StubLLMAdapter(summary="Four small events, summarised.")
    with Engram(path=":memory:", agent_id=_AGENT_ID, llm=stub, events_path=events_path) as mem:
        source_ids = mem.observe_many([ObserveInput(content=f"Event number {i}") for i in range(4)])
        for ep in mem._store.get_episodes_below_importance(1.1):
            mem._store.update_importance(ep.id, 0.1)
        result = mem.compress(max_episodes=0, importance_threshold=0.3, batch_size=4)

    assert result.episodes_removed == 4
    assert result.summaries_created == 1

    events = _read_ndjson(events_path)
    _assert_chain(events)
    forgotten = [e for e in events if e["type"] == "memory_forgotten"]
    assert len(forgotten) == result.episodes_removed
    assert {e["data"]["memory_id"] for e in forgotten} == set(source_ids)
    assert all(e["data"]["kind"] == "episodic" for e in forgotten)


def test_compress_names_the_summary_that_replaced_each_deleted_episode(tmp_path) -> None:
    """A compression run is a deletion and a creation in one operation, so the
    log has to say which summary each erased episode was folded into."""
    events_path = tmp_path / "events.ndjson"
    stub = StubLLMAdapter(summary="Four small events, summarised.")
    with Engram(path=":memory:", agent_id=_AGENT_ID, llm=stub, events_path=events_path) as mem:
        mem.observe_many([ObserveInput(content=f"Event number {i}") for i in range(4)])
        for ep in mem._store.get_episodes_below_importance(1.1):
            mem._store.update_importance(ep.id, 0.1)
        mem.compress(max_episodes=0, importance_threshold=0.3, batch_size=4)

    events = _read_ndjson(events_path)
    written = [e for e in events if e["type"] == "memory_written"]
    forgotten = [e for e in events if e["type"] == "memory_forgotten"]
    # 4 originals plus 1 summary written, and no more: a compression run must
    # not look like more creation than happened.
    assert len(written) == 5
    summary_id = written[-1]["data"]["memory_id"]
    assert {e["data"]["replaced_by"] for e in forgotten} == {summary_id}


# ------------------------------------------------------------------
# prev_hash chain (SPEC.md §6.5)
# ------------------------------------------------------------------

# Cross-language pinned vectors: agent-stack-go/event/testdata/chain-vectors.json
# is the normative cross-language truth (Go: event.Canonicalize/ChainHash; Rust:
# tokenfuse's agent-event exporter; here: canonicalize/chain_hash). Every
# implementation MUST reproduce these byte-for-byte. The vector events carry
# envelope keys (on_behalf_of, run_id) that engram's own emit() never sets
# itself -- canonicalize/chain_hash operate on plain dicts, so that is fine.

_VEC_EVENT_1 = {
    "schema": "taipanbox.dev/agent-event/v0.2",
    "ts": "2026-07-24T12:00:00Z",
    "source": "wardryx",
    "type": "policy_deny",
    "agent_id": "agent://acme.example/support/tier1-bot",
    "severity": "high",
    "run_id": "run-0001",
    "data": {"policy": "finance-guard", "reason": "deny_tool: shell"},
}
_VEC_CANONICAL_1 = (
    '{"agent_id":"agent://acme.example/support/tier1-bot","data":{"policy":"finance-guard",'
    '"reason":"deny_tool: shell"},"run_id":"run-0001","schema":"taipanbox.dev/agent-event/v0.2",'
    '"severity":"high","source":"wardryx","ts":"2026-07-24T12:00:00Z","type":"policy_deny"}'
)
_VEC_HASH_1 = "sha256:b43502c0ed6893238f2635be7a909cde89df1c2eecaef4d84871b83cf21cb31b"

_VEC_EVENT_2 = {
    "schema": "taipanbox.dev/agent-event/v0.2",
    "ts": "2026-07-24T12:00:01Z",
    "source": "tokenfuse",
    "type": "budget_exhausted",
    "agent_id": "agent://acme.example/support/tier1-bot",
    "severity": "critical",
    "run_id": "run-0001",
    "on_behalf_of": ["user://acme.example/alice", "agent://acme.example/orchestrator"],
    "data": {"budget_usd": 12.5, "n": 3, "note": "обмеження діє", "nested": {"b": 2, "a": 1}},
}
_VEC_CANONICAL_2 = (
    '{"agent_id":"agent://acme.example/support/tier1-bot","data":{"budget_usd":12.5,"n":3,'
    '"nested":{"a":1,"b":2},"note":"обмеження діє"},"on_behalf_of":["user://acme.example/alice",'
    '"agent://acme.example/orchestrator"],"run_id":"run-0001",'
    '"schema":"taipanbox.dev/agent-event/v0.2","severity":"critical","source":"tokenfuse",'
    '"ts":"2026-07-24T12:00:01Z","type":"budget_exhausted"}'
)
_VEC_HASH_2 = "sha256:488f1017967bf9510c62d7c31b9d5a0086ff2000d90a7d4266f171a131430243"

_VEC_EVENT_3 = {
    "schema": "taipanbox.dev/agent-event/v0.2",
    "ts": "2026-07-24T12:00:02Z",
    "source": "qryx",
    "type": "evidence_signed",
    "agent_id": "agent://acme.example/support/tier1-bot",
    "severity": "info",
    "data": {"algo": "ML-DSA-87"},
}
_VEC_CANONICAL_3 = (
    '{"agent_id":"agent://acme.example/support/tier1-bot","data":{"algo":"ML-DSA-87"},'
    '"schema":"taipanbox.dev/agent-event/v0.2","severity":"info","source":"qryx",'
    '"ts":"2026-07-24T12:00:02Z","type":"evidence_signed"}'
)
_VEC_HASH_3 = "sha256:998cbc146b07e115318ce378e0579fcd1927066ef4316900ec7d66ba157e7c4b"

_CHAIN_VECTORS = [
    (_VEC_EVENT_1, _VEC_CANONICAL_1, _VEC_HASH_1),
    (_VEC_EVENT_2, _VEC_CANONICAL_2, _VEC_HASH_2),
    (_VEC_EVENT_3, _VEC_CANONICAL_3, _VEC_HASH_3),
]


def test_canonicalize_and_chain_hash_match_pinned_vectors() -> None:
    """engram.events.canonicalize/chain_hash MUST reproduce the cross-language
    vectors byte-for-byte -- see the comment above _CHAIN_VECTORS."""
    for event, canonical, expected_hash in _CHAIN_VECTORS:
        assert canonicalize(event) == canonical.encode("utf-8")
        assert chain_hash(event) == expected_hash


def test_emit_chains_two_events(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)
    log.emit("memory_written", _AGENT_ID, {"memory_id": "a", "kind": "episodic"})
    log.emit("memory_written", _AGENT_ID, {"memory_id": "b", "kind": "episodic"})

    events = _read_ndjson(events_path)
    assert len(events) == 2
    assert "prev_hash" not in events[0]
    assert events[1]["prev_hash"] == chain_hash(events[0])
    for event in events:
        _validate(event)


def test_reopened_event_log_resumes_the_chain(tmp_path) -> None:
    """One file, one chain: a new EventLog over an existing file continues
    the chain rather than restarting it (SPEC.md §6.5)."""
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)
    log.emit("memory_written", _AGENT_ID, {"memory_id": "a", "kind": "episodic"})
    log.emit("memory_written", _AGENT_ID, {"memory_id": "b", "kind": "episodic"})

    resumed = EventLog(events_path)
    resumed.emit("memory_written", _AGENT_ID, {"memory_id": "c", "kind": "episodic"})

    events = _read_ndjson(events_path)
    assert len(events) == 3
    assert events[2]["prev_hash"] == chain_hash(events[1])


def test_malformed_tail_starts_a_fresh_chain(tmp_path) -> None:
    """A tail that does not parse as JSON is exactly like no file at all:
    EventLog starts a fresh chain rather than raising (fail-open)."""
    events_path = tmp_path / "events.ndjson"
    events_path.write_text("{not json at all\n")

    log = EventLog(events_path)
    log.emit("memory_written", _AGENT_ID, {"memory_id": "a", "kind": "episodic"})

    lines = [line for line in events_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    new_event = json.loads(lines[1])
    assert "prev_hash" not in new_event


# ------------------------------------------------------------------
# agent_id shape at the seam
# ------------------------------------------------------------------
#
# The defect, found in a read-only audit on 2026-08-05: the envelope's own
# schema requires agent_id to match ^agent://[a-z0-9.-]+/[a-z0-9._/-]+$ and
# emit() accepted any non-empty string and wrote it verbatim, while the
# README, the CLI and engram-mcp all demonstrated ids that cannot validate.
# The lines are rejected where they are consumed, which is somewhere else,
# by somebody else, silently.


def test_a_nonconforming_agent_id_is_still_written(tmp_path, caplog) -> None:
    """The event is not dropped. Refusing to emit would turn a wire-contract
    mismatch into an empty log for exactly the caller who needs to see it,
    which is a worse failure than a line a consumer rejects."""
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)

    with caplog.at_level(logging.WARNING, logger="engram.events"):
        log.emit("memory_written", "planner", {"memory_id": "a", "kind": "episodic"})

    events = _read_ndjson(events_path)
    assert len(events) == 1
    assert events[0]["agent_id"] == "planner"


def test_a_nonconforming_agent_id_is_warned_about_and_counted(tmp_path, caplog) -> None:
    """Emitting it quietly is what made this invisible. The warning names the
    id and the grammar, and a counter carries the number for callers."""
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)

    with caplog.at_level(logging.WARNING, logger="engram.events"):
        log.emit("memory_written", "planner", {"memory_id": "a", "kind": "episodic"})

    assert log.nonconforming_agent_id == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "a non-conforming agent_id was written with nothing said about it"
    said = "\n".join(r.getMessage() for r in warnings)
    assert "planner" in said
    assert "agent://" in said


def test_the_warning_is_said_once_per_id_and_counted_every_time(tmp_path, caplog) -> None:
    """A write loop must not turn one misconfigured id into a log flood, and
    the count must still be the truth about how many lines are affected."""
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)

    with caplog.at_level(logging.WARNING, logger="engram.events"):
        for i in range(5):
            log.emit("memory_written", "planner", {"memory_id": str(i), "kind": "episodic"})

    assert log.nonconforming_agent_id == 5
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_emit_many_counts_every_item_it_wrote(tmp_path, caplog) -> None:
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)

    with caplog.at_level(logging.WARNING, logger="engram.events"):
        log.emit_many(
            "memory_written",
            "planner",
            [{"memory_id": str(i), "kind": "episodic"} for i in range(3)],
        )

    assert log.nonconforming_agent_id == 3
    assert len(_read_ndjson(events_path)) == 3


def test_a_canonical_agent_id_is_neither_warned_about_nor_counted(tmp_path, caplog) -> None:
    """The guard against a fix that shouts at everybody. This one passed before
    the change too: nothing was ever wrong with the conforming path."""
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)

    with caplog.at_level(logging.WARNING, logger="engram.events"):
        log.emit("memory_written", _AGENT_ID, {"memory_id": "a", "kind": "episodic"})

    assert log.nonconforming_agent_id == 0
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_the_grammar_here_is_the_one_in_the_schema(tmp_path) -> None:
    """Two copies of a pattern drift. This one is written down twice on
    purpose (the vendored schema is the contract, the module needs it at
    runtime), so the test holds them equal rather than trusting them to be."""
    from engram.events import AGENT_ID_MAX_LENGTH, AGENT_ID_PATTERN

    schema_agent_id = _SCHEMA["properties"]["agent_id"]
    assert AGENT_ID_PATTERN.pattern == schema_agent_id["pattern"]
    assert schema_agent_id["maxLength"] == AGENT_ID_MAX_LENGTH


def test_a_nonconforming_line_is_rejected_by_the_schema_it_claims_to_speak(tmp_path) -> None:
    """What the warning is warning about, stated as the consumer states it.
    The chain is unaffected: the line is well formed, its agent_id is not."""
    events_path = tmp_path / "events.ndjson"
    log = EventLog(events_path)
    log.emit("memory_written", "planner", {"memory_id": "a", "kind": "episodic"})

    event = _read_ndjson(events_path)[0]
    with pytest.raises(jsonschema.ValidationError):
        _validate(event)
    assert chain_hash(event).startswith("sha256:")


def test_every_agent_id_this_project_documents_can_validate() -> None:
    """The examples are the defect's other half: a reader who copies one gets
    a store that works and an event log nothing downstream will accept.

    Scans the documented ids rather than trusting a sweep to have caught them
    all, so adding a new example with a bare name fails here rather than in
    somebody's ingest.
    """
    from engram.events import is_canonical_agent_id

    # Quoted literals only, so `agent_id=None`, `agent_id=key` and
    # `agent_id=getattr(...)` are not mistaken for examples; and command lines
    # minus their metavars, so `[--agent-id ID]` in a usage string is not read
    # as somebody's actual id.
    literals = re.compile(r"""(?<![\w.])agent_id\s*=\s*["']([^"']+)["']""")
    # An id, not the next word of a sentence about the flag: the character set
    # here is deliberately wider than the grammar under test, so a bare
    # "planner" is caught while "(overrides --agent-id scope)" is not.
    command_lines = re.compile(r"--agent-id[ =]([A-Za-z0-9:/._-]+)(?=[\s,.\]]|$)")
    documented = (
        "README.md",
        "GETTING_STARTED.md",
        "docs/api-reference.md",
        "engram/cli.py",
        "engram/mcp_server.py",
    )

    offenders: list[str] = []
    for doc in documented:
        path = Path(doc)
        if not path.exists():
            continue
        text = path.read_text()
        found = literals.findall(text) + [c for c in command_lines.findall(text) if not c.isupper()]
        for candidate in found:
            if not is_canonical_agent_id(candidate):
                offenders.append(f"{doc}: {candidate}")
    assert not offenders, (
        "these documented agent ids cannot validate against the envelope schema, "
        "so copying one produces an event log a consumer rejects: " + "; ".join(offenders)
    )
