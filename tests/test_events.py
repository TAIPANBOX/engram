"""Tests for engram.events and its wiring into Engram's write/reflect paths.

Schema validation uses a vendored copy of the Agent Passport
``agent-event.schema.json`` (see ``tests/fixtures/agent-event.schema.json``,
copied from ``TAIPANBOX/agent-passport`` -- SPEC.md §6). Vendored rather than
fetched at test time: CI checks out only this repo, and validating a wire
contract should never depend on a live network call.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import jsonschema
import pytest

from engram import Engram, StubLLMAdapter
from engram.events import EventLog, resolve_events_path

_SCHEMA_PATH = Path(__file__).parent / "fixtures" / "agent-event.schema.json"
_SCHEMA = json.loads(_SCHEMA_PATH.read_text())

_AGENT_ID = "agent://acme-bank.example/support/tier1-bot"


def _read_ndjson(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate(event: dict[str, object]) -> None:
    jsonschema.validate(instance=event, schema=_SCHEMA)


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


def test_no_llm_reflect_emits_no_contradiction_event(tmp_path) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=_AGENT_ID, events_path=events_path) as mem:
        mem.observe("Something happened today")
        mem.reflect()

    events = _read_ndjson(events_path)
    assert all(e["type"] != "contradiction_found" for e in events)
