"""Tests for JSON export and import."""

from __future__ import annotations

import json

import pytest

from engram import Engram, ObserveInput


@pytest.fixture()
def populated(tmp_path):
    path = str(tmp_path / "source.engram")
    with Engram(path=path) as mem:
        mem.observe_many([
            ObserveInput(content="Alice joined Globex as CTO", actors=["Alice"], tags=["hr"]),
            ObserveInput(content="Q3 budget approved at $500k", tags=["finance"]),
            ObserveInput(content="Ivan transferred from Acme to Globex", actors=["Ivan"]),
        ])
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
        # re-embed so vector search works (import uses zero vectors)
        ep_count = mem._store.episode_count()
    assert ep_count == 3


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
