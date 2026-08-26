"""Tests for engram.events and its wiring into Engram's write/reflect paths.

Schema validation uses a vendored copy of the Agent Passport
``agent-event.v0.2.schema.json`` (see
``tests/fixtures/agent-event.v0.2.schema.json``, copied byte for byte from
``TAIPANBOX/agent-passport`` -- SPEC.md §6). Vendored rather than fetched at
test time: CI checks out only this repo, and validating a wire contract should
never depend on a live network call.

The copy carries the canonical file's NAME as well as its bytes, because
agent-passport owns two of them: ``agent-event.schema.json`` is v0.1 and
``agent-event.v0.2.schema.json`` is v0.2. A copy of the second under the name
of the first is a file that can never be compared with its original.
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
from engram.events import SCHEMA, EventLog, canonicalize, chain_hash, resolve_events_path

_SCHEMA_PATH = Path(__file__).parent / "fixtures" / "agent-event.v0.2.schema.json"
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
    assert event["schema"] == "taipanbox.dev/agent-event/v0.2"
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
# The envelope version this build speaks (SPEC.md §6.4)
# ------------------------------------------------------------------
#
# Engram was the last emitter on v0.1; heraldyx, agent-stack-go, mockryx,
# verdryx and genaryx were already on v0.2. SPEC.md §6.4 permits an existing
# emitter to stay on v0.1 forever, so this is not a violation being fixed. It
# is one dialect instead of two.

#: v0.1's ``agent_id`` constraints, as literals. v0.1 is published and frozen,
#: and this repo no longer vendors it, so the only honest way to compare the
#: two versions is to write the older one down.
_V0_1_AGENT_ID_PATTERN = "^agent://[a-z0-9.-]+/[a-z0-9._/-]+$"
_V0_1_AGENT_ID_MAX_LENGTH = 255


def test_the_vendored_schema_is_the_v0_2_contract_engram_now_speaks() -> None:
    """The module constant and the contract file are two statements of one
    fact, so hold them equal rather than trusting a reader to notice."""
    assert SCHEMA == "taipanbox.dev/agent-event/v0.2"
    assert _SCHEMA["properties"]["schema"]["const"] == SCHEMA
    assert "/v0.2/" in _SCHEMA["$id"]
    # Two fields separate the versions, and each names a different way for a
    # vendored copy to be the wrong file. v0.1 closed `source` to four names
    # and v0.2 opens it (SPEC.md §6.4), so a copy still carrying the enum is a
    # v0.1 schema wearing a v0.2 const. v0.2 also carries `delegation_proof`
    # (SPEC.md §5.2), so a copy without it is a v0.2 file taken before
    # agent-passport added the field, which is a version of the contract that
    # accepts a proved chain and a proof it will not read.
    assert "enum" not in _SCHEMA["properties"]["source"]
    assert "delegation_proof" in _SCHEMA["properties"]


def test_the_agent_id_rule_is_the_same_under_v0_2_as_it_was_under_v0_1() -> None:
    """Why the version move does not reopen the warn-rather-than-refuse
    decision. That decision was made against a rule, and this is the check
    that the rule did not move underneath it: an id v0.1 rejected, v0.2
    rejects, by the same pattern and the same cap. A later version that
    tightens either fails here, which is where it should be argued again.
    """
    agent_id = _SCHEMA["properties"]["agent_id"]
    assert agent_id["pattern"] == _V0_1_AGENT_ID_PATTERN
    assert agent_id["maxLength"] == _V0_1_AGENT_ID_MAX_LENGTH


def test_a_delegation_proof_is_accepted_but_engram_never_writes_one(tmp_path) -> None:
    """SPEC.md §5.2, and the two halves of it this repo actually holds.

    ACCEPTED, and enforced rather than waved through. The envelope's own
    ``additionalProperties`` is ``true``, so a schema that had never heard of
    ``delegation_proof`` accepts any object at all under that name. A check
    that only validated a well-formed proof would therefore pass against the
    copy this one replaced, and prove nothing about either.

    NEVER WRITTEN. ``_envelope`` writes a fixed set of keys and
    ``on_behalf_of`` is not among them, so there is no delegation chain here to
    prove and nothing to attach a proof to. Absent means NOT proven, which is
    the honest reading of every line this module has ever written.
    """
    proved = {
        "schema": SCHEMA,
        "ts": "2026-08-26T12:00:00Z",
        "source": "engram",
        "type": "memory_written",
        "agent_id": _AGENT_ID,
        "severity": "info",
        "on_behalf_of": ["user://acme-bank.example/alice"],
        "delegation_proof": {
            "jti": "01K3S4V6QZ0000000000000000",
            "jkt": "NzbLsXh8uDCcd-6MNwXF4W_7noWXFZAfHkxZsRGC9Xs",
            "iss": "https://idryx.acme-bank.example",
            "exp": 1787836800,
        },
        "data": {"memory_id": "m-1", "kind": "episodic"},
    }
    _validate(proved)

    # A proof missing the key binding is the one that matters: without `jkt` it
    # names a token but not who was holding it, which is the whole thing a
    # chain of names cannot say.
    unbound = json.loads(json.dumps(proved))
    del unbound["delegation_proof"]["jkt"]
    with pytest.raises(jsonschema.ValidationError):
        _validate(unbound)

    # And the live credential may not ride along inside the record: the proof
    # is closed, so a token smuggled in beside it is rejected here rather than
    # replicated down a hash-chained stream.
    carrying_the_token = json.loads(json.dumps(proved))
    carrying_the_token["delegation_proof"]["token"] = "eyJhbGciOiJFZERTQSJ9.e30.sig"
    with pytest.raises(jsonschema.ValidationError):
        _validate(carrying_the_token)

    events_path = tmp_path / "events.ndjson"
    EventLog(events_path).emit("memory_written", _AGENT_ID, {"memory_id": "a", "kind": "episodic"})
    written = _read_ndjson(events_path)[0]
    _validate(written)
    assert "delegation_proof" not in written
    assert "on_behalf_of" not in written


def test_every_envelope_version_this_project_documents_is_the_one_it_emits() -> None:
    """A consumer reads the docs to learn which version arrives, and SPEC.md
    §6.4 makes that a real question rather than a formality: both versions are
    valid on the wire, so a stale sentence here is not obviously wrong to
    anybody. It fails on the next bump instead of in somebody's ingest.

    Refuses when it finds nothing, because a check that goes green once its
    subject has vanished is worse than no check at all.

    A sentence about an OLDER version names it as v0.1 rather than spelling the
    whole identifier out, so the only full ``taipanbox.dev/agent-event/...``
    string in the docs stays the one this build actually writes.
    """
    stated = re.compile(r"taipanbox\.dev/agent-event/v[0-9.]+")
    documented = ("README.md", "docs/api-reference.md", "GETTING_STARTED.md", "DATA_FLOW.md")

    found: list[str] = []
    offenders: list[str] = []
    for doc in documented:
        path = Path(doc)
        if not path.exists():
            continue
        for version in stated.findall(path.read_text()):
            found.append(f"{doc}: {version}")
            if version != SCHEMA:
                offenders.append(f"{doc}: {version}")

    assert found, (
        "no documented file names the envelope version, so this check measured "
        f"nothing. One of {', '.join(documented)} must say which version a "
        "consumer receives."
    )
    assert not offenders, (
        f"these say engram emits an envelope version it does not emit ({SCHEMA}): "
        + "; ".join(offenders)
    )


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

# Vector 4. `delegation_proof` (SPEC 5.2) is a top-level sibling of `data`, so
# it is exactly the member a language's event STRUCT may have no field for.
# Python is safe by construction here, because these functions take a dict and
# a dict carries whatever the line carried, and this vector is what turns "safe
# by construction" into "proved on every run". Go's event type was not: it
# hashed a re-marshal of its own struct until 2026-08-26, the member vanished
# before the digest, and an honestly chained stream was reported as tampered
# with by our own conformance tool.
_VEC_EVENT_4 = {
    "schema": "taipanbox.dev/agent-event/v0.3",
    "ts": "2026-08-26T12:00:03Z",
    "source": "vouchryx",
    "type": "delegation_issued",
    "agent_id": "agent://acme.example/support/tier1-bot",
    "severity": "info",
    "run_id": "run-0001",
    "delegation_proof": {
        "jti": "tok-9f2c",
        "jkt": "NzbLsXh8uDCcd-6MNwXF4W_7noWXFZAfHkxZsRGC9Xs",
        "iss": "https://idryx.acme.example",
        "exp": 1786000000,
    },
    "data": {"scope": "read:tickets"},
}
_VEC_CANONICAL_4 = (
    '{"agent_id":"agent://acme.example/support/tier1-bot","data":{"scope":"read:tickets"},'
    '"delegation_proof":{"exp":1786000000,"iss":"https://idryx.acme.example",'
    '"jkt":"NzbLsXh8uDCcd-6MNwXF4W_7noWXFZAfHkxZsRGC9Xs","jti":"tok-9f2c"},'
    '"run_id":"run-0001","schema":"taipanbox.dev/agent-event/v0.3","severity":"info",'
    '"source":"vouchryx","ts":"2026-08-26T12:00:03Z","type":"delegation_issued"}'
)
_VEC_HASH_4 = "sha256:97161b1b4dd0b64d683e27611279beb7024a91d0dba2fd736d10e96edabd7680"

_CHAIN_VECTORS = [
    (_VEC_EVENT_1, _VEC_CANONICAL_1, _VEC_HASH_1),
    (_VEC_EVENT_2, _VEC_CANONICAL_2, _VEC_HASH_2),
    (_VEC_EVENT_3, _VEC_CANONICAL_3, _VEC_HASH_3),
    (_VEC_EVENT_4, _VEC_CANONICAL_4, _VEC_HASH_4),
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


def test_a_chain_written_under_v0_1_continues_under_v0_2(tmp_path) -> None:
    """The upgrade case, and the reason it is a continuation rather than a
    restart.

    ``schema`` sits inside the envelope, so it sits inside the canonical bytes
    and therefore inside the hash. What it does NOT sit inside is the rule:
    line N's ``prev_hash`` is the hash of line N-1 whatever either line
    declares, so the link across the version boundary verifies like any other.

    Restarting instead would mean writing a head line at exactly the point an
    upgrade happened, and a head line is where a verifier stops being able to
    tell a legal restart from a truncation (agent-stack-go's ``VerifyChain``
    reports them separately and calls neither a break). Deleting evidence at
    the one moment somebody might want it is a strange way to keep an audit
    log, and nothing is gained: the old lines stay valid, v0.1 and v0.2 are
    both accepted by every consumer in the estate (SPEC.md §6.4), and one file
    stays one chain.

    The v0.1 line is deliberately not schema-validated here. This repo vendors
    only the version it emits, and validating the older half of a mixed file is
    the consumer's job, which is what makes it a consumer's job to accept both.
    """
    events_path = tmp_path / "events.ndjson"
    v0_1_line = {
        "schema": "taipanbox.dev/agent-event/v0.1",
        "ts": "2026-08-06T12:00:00.000Z",
        "source": "engram",
        "type": "memory_written",
        "severity": "info",
        "agent_id": _AGENT_ID,
        "data": {"memory_id": "written-before-the-upgrade", "kind": "episodic"},
    }
    events_path.write_text(json.dumps(v0_1_line, separators=(",", ":")) + "\n")

    EventLog(events_path).emit("memory_written", _AGENT_ID, {"memory_id": "b", "kind": "episodic"})

    events = _read_ndjson(events_path)
    assert len(events) == 2
    assert events[0]["schema"] == "taipanbox.dev/agent-event/v0.1"
    assert events[1]["schema"] == "taipanbox.dev/agent-event/v0.2"
    _validate(events[1])
    # A continuation, not a restart: the new line carries a prev_hash at all,
    # and it is the hash of the v0.1 line that precedes it.
    assert "prev_hash" in events[1]
    assert events[1]["prev_hash"] == chain_hash(events[0])
    assert events[1]["prev_hash"] == chain_hash(v0_1_line)


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
