"""A person's own fact outranks a model's.

The behaviour these tests hold is written as scenarios in
``features/human-priority-and-refutation.feature``, one test per scenario, and
``scripts/features-are-bound.sh`` keeps the two in step. Each test was run
against the unchanged code first and failed there.

The rule, in one paragraph: a fact whose provenance is ``user`` and that was
not written quarantined is the person's own. Nothing with a lower provenance
closes it: not a reflection pass, and not a fact asserted beside it. The
attempt is not hidden. The newer fact is recorded next to the person's one,
``contradictions()`` lists the pair, and a ``contradiction_found`` event names
both. Between two facts that are both the person's own, recency still decides,
exactly as it always did.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engram import Engram, StubLLMAdapter

AGENT = "agent://acme.example/assistant"


def _stub(obj: str) -> StubLLMAdapter:
    return StubLLMAdapter(
        facts=[{"subject": "Ivan", "predicate": "works_at", "object": obj, "confidence": 0.9}]
    )


def _reflect(
    mem: Engram, obj: str, text: str, *, provenance: str, quarantined: bool = False
) -> tuple[str, Any]:
    """Observe *text* with the given source, then reflect with a model that extracts *obj*."""
    mem._llm = _stub(obj)
    episode_id = mem.observe(text, provenance=provenance, quarantined=quarantined)
    return episode_id, mem.reflect()


def _active(mem: Engram) -> dict[str, Any]:
    return {f.object: f for f in mem._store.get_active_facts("Ivan", "works_at")}


def _contradiction_events(path: Path) -> list[dict[str, Any]]:
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [e for e in lines if e["type"] == "contradiction_found"]


def test_a_models_extraction_cannot_close_a_persons_fact(tmp_path: Path) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=AGENT, events_path=events_path) as mem:
        _reflect(mem, "Globex", "I work at Globex", provenance="user")
        person = _active(mem)["Globex"]
        assert mem.why(person.id)["provenance"] == "user"

        _, run = _reflect(
            mem, "Initech", "A web page says Ivan works at Initech", provenance="external"
        )

        active = _active(mem)
        assert set(active) == {"Globex", "Initech"}
        held = mem._store.get_fact(person.id)
        assert held is not None
        assert held.valid_to is None
        assert held.superseded_at is None
        assert held.superseded_by is None
        assert run.contradictions_resolved == 0
        pairs = mem.contradictions()
        assert len(pairs) == 1
        assert {f.object for f in pairs[0]} == {"Globex", "Initech"}

    events = _contradiction_events(events_path)
    assert len(events) == 1
    assert events[0]["severity"] == "medium"
    assert events[0]["data"] == {
        "memory_id": active["Initech"].id,
        "conflicting_memory_id": person.id,
    }


def test_a_persons_newer_word_still_replaces_their_older_word() -> None:
    with Engram(path=":memory:") as mem:
        _reflect(mem, "Globex", "I work at Globex", provenance="user")
        _, run = _reflect(mem, "Initech", "I moved to Initech", provenance="user")
        assert set(_active(mem)) == {"Initech"}
        assert run.contradictions_resolved == 1


def test_a_quarantined_write_does_not_count_as_the_persons_word() -> None:
    # On the incoming side: a "user" episode written after untrusted reading
    # cannot close a pristine person's fact.
    with Engram(path=":memory:") as mem:
        _reflect(mem, "Globex", "I work at Globex", provenance="user")
        _, run = _reflect(mem, "Initech", "I work at Initech", provenance="user", quarantined=True)
        assert set(_active(mem)) == {"Globex", "Initech"}
        assert run.contradictions_resolved == 0

    # On the held side: a fact that was itself extracted from a quarantined
    # episode is not the person's own, so a model may replace it.
    with Engram(path=":memory:") as mem:
        _reflect(mem, "Globex", "I work at Globex", provenance="user", quarantined=True)
        _, run = _reflect(mem, "Initech", "A page says Initech", provenance="external")
        assert set(_active(mem)) == {"Initech"}
        assert run.contradictions_resolved == 1


def test_the_protection_outlives_the_episodes_a_fact_was_extracted_from() -> None:
    with Engram(path=":memory:") as mem:
        source_id, _ = _reflect(mem, "Globex", "I work at Globex", provenance="user")
        person = _active(mem)["Globex"]

        mem.forget(source_id)
        assert mem._store.get_episodes_by_ids(person.derived_from) == []
        assert mem.why(person.id)["provenance"] == "user"

        _reflect(mem, "Initech", "A page says Initech", provenance="external")
        assert set(_active(mem)) == {"Globex", "Initech"}
        held = mem._store.get_fact(person.id)
        assert held is not None and held.valid_to is None


def test_a_models_asserted_fact_leaves_a_persons_fact_in_place_and_is_reported(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=AGENT, events_path=events_path) as mem:
        person = mem.assert_fact("Ivan", "works_at", "Globex", provenance="user")
        model = mem.assert_fact("Ivan", "works_at", "Initech")

        # Agreement, and a second person's word, raise nothing.
        mem.assert_fact("Ivan", "works_at", "Globex")
        mem.assert_fact("Ivan", "works_at", "Umbrella", provenance="user")

        held = mem._store.get_fact(person)
        assert held is not None
        assert held.valid_to is None and held.superseded_at is None
        assert held.provenance == "user"
        assert mem.why(person)["provenance"] == "user"
        assert mem.why(model)["provenance"] == "unknown"

    events = _contradiction_events(events_path)
    assert [e["data"] for e in events] == [
        {"memory_id": model, "conflicting_memory_id": person},
    ]


def test_a_fact_that_cites_a_persons_episode_is_not_thereby_the_persons(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "events.ndjson"
    with Engram(path=":memory:", agent_id=AGENT, events_path=events_path) as mem:
        said = mem.observe("I work at Globex", provenance="user")
        cited = mem.assert_fact("Ivan", "works_at", "Globex", source=said)
        assert mem.why(cited)["provenance"] == "unknown"

        mem.assert_fact("Ivan", "works_at", "Initech")

    assert _contradiction_events(events_path) == []
