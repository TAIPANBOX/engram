"""Tests for JSON export and import."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from engram import Engram, ObserveInput


@pytest.fixture()
def populated(tmp_path):
    path = str(tmp_path / "source.engram")
    with Engram(path=path) as mem:
        mem.observe_many(
            [
                ObserveInput(content="Alice joined Globex as CTO", actors=["Alice"], tags=["hr"]),
                ObserveInput(content="Q3 budget approved at $500k", tags=["finance"]),
                ObserveInput(content="Ivan transferred from Acme to Globex", actors=["Ivan"]),
            ]
        )
        mem.assert_fact("Alice", "role", "CTO", confidence=0.95)
        mem.assert_fact("Ivan", "works_at", "Globex", confidence=0.9)
    return path


# ------------------------------------------------------------------
# export_json
# ------------------------------------------------------------------


def test_export_creates_file(populated, tmp_path):
    dest = str(tmp_path / "dump.json")
    with Engram(path=populated) as mem:
        mem.export_json(dest)
    assert (tmp_path / "dump.json").exists()


def test_export_counts_match(populated, tmp_path):
    dest = str(tmp_path / "dump.json")
    with Engram(path=populated) as mem:
        doc = mem.export_json(dest)
    assert doc["counts"]["episodes"] == 3
    assert doc["counts"]["facts"] == 2


def test_export_valid_json(populated, tmp_path):
    dest = str(tmp_path / "dump.json")
    with Engram(path=populated) as mem:
        mem.export_json(dest)
    doc = json.loads((tmp_path / "dump.json").read_text())
    assert doc["engram_export_version"] == 1
    assert "exported_at" in doc
    assert len(doc["episodes"]) == 3


def test_export_episode_fields(populated, tmp_path):
    dest = str(tmp_path / "dump.json")
    with Engram(path=populated) as mem:
        doc = mem.export_json(dest)
    ep = doc["episodes"][0]
    assert "id" in ep
    assert "content" in ep
    assert "timestamp" in ep
    assert "actors" in ep


def test_export_fact_fields(populated, tmp_path):
    dest = str(tmp_path / "dump.json")
    with Engram(path=populated) as mem:
        doc = mem.export_json(dest)
    fact = doc["facts"][0]
    assert "subject" in fact
    assert "predicate" in fact
    assert "object" in fact
    assert "valid_from" in fact
    assert "confidence" in fact


# ------------------------------------------------------------------
# import_json — fresh store
# ------------------------------------------------------------------


def test_import_restores_episodes(populated, tmp_path):
    dest = str(tmp_path / "dump.json")
    target = str(tmp_path / "target.engram")
    with Engram(path=populated) as mem:
        mem.export_json(dest)
    with Engram(path=target) as mem:
        counts = mem.import_json(dest)
    assert counts["episodes"] == 3
    assert counts["facts"] == 2


def test_import_episodes_are_recalled(populated, tmp_path):
    dest = str(tmp_path / "dump.json")
    target = str(tmp_path / "target.engram")
    with Engram(path=populated) as mem:
        mem.export_json(dest)
    with Engram(path=target) as mem:
        mem.import_json(dest)
        ep_count = mem._store.episode_count()
        # Episodes are re-embedded on import, so vector recall must actually
        # surface the right one (regression: import used to write zero vectors).
        results = mem.recall("who became CTO at Globex?", k=1)
    assert ep_count == 3
    assert results
    assert "Globex" in results[0].episode.content


def test_import_episodes_are_recalled_via_hybrid(populated, tmp_path):
    """Imported episodes must also be visible to the FTS/BM25 half of recall."""
    dest = str(tmp_path / "dump.json")
    target = str(tmp_path / "target.engram")
    with Engram(path=populated) as mem:
        mem.export_json(dest)
    with Engram(path=target) as mem:
        mem.import_json(dest)
        results = mem.recall("budget", k=3, mode="hybrid")
    assert any("budget" in r.episode.content.lower() for r in results)


def test_import_facts_available(populated, tmp_path):
    dest = str(tmp_path / "dump.json")
    target = str(tmp_path / "target.engram")
    with Engram(path=populated) as mem:
        mem.export_json(dest)
    with Engram(path=target) as mem:
        mem.import_json(dest)
        facts = mem.timeline("Alice")
    assert len(facts) == 1
    assert facts[0].object == "CTO"


# ------------------------------------------------------------------
# import_json — merge mode
# ------------------------------------------------------------------


def test_import_merge_skips_duplicates(populated, tmp_path):
    dest = str(tmp_path / "dump.json")
    with Engram(path=populated) as mem:
        mem.export_json(dest)
        # import into the same store — all ids already exist
        counts = mem.import_json(dest, merge=True)
    assert counts["episodes"] == 0
    assert counts["facts"] == 0


def test_import_no_merge_raises_on_duplicate(populated, tmp_path):
    dest = str(tmp_path / "dump.json")
    with Engram(path=populated) as mem:
        mem.export_json(dest)
        with pytest.raises(Exception, match=r"UNIQUE|ABORT|already"):
            mem.import_json(dest, merge=False)


# ------------------------------------------------------------------
# Round-trip
# ------------------------------------------------------------------


def test_round_trip_episode_count(populated, tmp_path):
    dest = str(tmp_path / "dump.json")
    target = str(tmp_path / "rt.engram")
    with Engram(path=populated) as src:
        src.export_json(dest)
    with Engram(path=target) as dst:
        dst.import_json(dest)
        assert dst._store.episode_count() == 3
        assert dst._store.fact_count() == 2


def test_export_empty_store(tmp_path):
    path = str(tmp_path / "empty.engram")
    dest = str(tmp_path / "empty.json")
    with Engram(path=path) as mem:
        doc = mem.export_json(dest)
    assert doc["counts"]["episodes"] == 0
    assert doc["counts"]["facts"] == 0


# ------------------------------------------------------------------
# export_json is scoped to the instance's agent_id
#
# Every other read path was scoped in 2.2.0 / 2.2.1 (get_episode,
# delete_episode, recall, prune_episodes, decay, get_neighbors). Export was
# the one left, so a scoped instance could dump another agent's raw episode
# text into a plaintext JSON file.
# ------------------------------------------------------------------


@pytest.fixture()
def two_agent_store(tmp_path):
    """A shared file holding one episode per agent, plus one edge each."""
    path = str(tmp_path / "team.engram")
    now = datetime.now(tz=UTC)
    with (
        Engram(path=path, agent_id="coder") as coder,
        Engram(path=path, agent_id="planner") as planner,
    ):
        coder.observe("coder wrote the retry loop", actors=["Ivan"])
        planner.observe("planner drafted the Q3 roadmap", actors=["Ivan"])
        coder.assert_fact("Ivan", "role", "engineer")
        planner.assert_fact("Ivan", "team", "platform")
        entity = coder._store.find_or_create_entity("Ivan", "person", now)
        coder._store.insert_edge("ep-coder", entity.id, "mentions", weight=1.0, created_at=now)
        planner._store.insert_edge("ep-planner", entity.id, "mentions", weight=1.0, created_at=now)
    return path


def test_scoped_export_contains_only_its_own_episodes(two_agent_store, tmp_path):
    dest = tmp_path / "coder.json"
    with Engram(path=two_agent_store, agent_id="coder") as coder:
        doc = coder.export_json(str(dest))

    assert doc["counts"]["episodes"] == 1
    assert [ep["agent_id"] for ep in doc["episodes"]] == ["coder"]
    # The point of the defect is plaintext leakage, so check the bytes on disk
    # rather than only the returned document.
    assert "roadmap" not in dest.read_text()


def test_scoped_export_contains_only_its_own_edges(two_agent_store, tmp_path):
    dest = str(tmp_path / "coder.json")
    with Engram(path=two_agent_store, agent_id="coder") as coder:
        doc = coder.export_json(dest)

    assert doc["counts"]["edges"] == 1
    assert [edge["src_id"] for edge in doc["edges"]] == ["ep-coder"]


def test_scoped_export_keeps_facts_and_entities_shared(two_agent_store, tmp_path):
    """Facts and entities are shared across agents by design (CHANGELOG 2.2.0
    and 2.2.1), so scoping export must not quietly turn that into a bug."""
    dest = str(tmp_path / "coder.json")
    with Engram(path=two_agent_store, agent_id="coder") as coder:
        doc = coder.export_json(dest)

    assert doc["counts"]["facts"] == 2
    assert doc["counts"]["entities"] == 1


def test_unscoped_export_still_contains_every_agents_data(two_agent_store, tmp_path):
    """The documented whole-store export is unchanged for an unscoped instance."""
    dest = str(tmp_path / "all.json")
    with Engram(path=two_agent_store) as everything:
        doc = everything.export_json(dest)

    assert doc["counts"]["episodes"] == 2
    assert doc["counts"]["edges"] == 2
    assert {ep["agent_id"] for ep in doc["episodes"]} == {"coder", "planner"}


def test_scoped_export_round_trips_into_a_fresh_store(two_agent_store, tmp_path):
    dest = str(tmp_path / "coder.json")
    target = str(tmp_path / "coder-copy.engram")
    with Engram(path=two_agent_store, agent_id="coder") as coder:
        coder.export_json(dest)
    with Engram(path=target, agent_id="coder") as restored:
        counts = restored.import_json(dest)
        assert counts["episodes"] == 1
        assert restored._store.episode_count() == 1
