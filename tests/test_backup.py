"""Tests for mem.backup() hot-backup API."""

from __future__ import annotations

import sqlite3

import pytest

from engram import Engram, ObserveInput


@pytest.fixture()
def populated(tmp_path):
    path = str(tmp_path / "source.engram")
    with Engram(path=path) as mem:
        mem.observe_many(
            [
                ObserveInput(content="Alice joined Globex"),
                ObserveInput(content="Q3 budget approved"),
            ]
        )
        mem.assert_fact("Alice", "role", "CTO")
    return path


def test_backup_creates_file(populated, tmp_path):
    dest = str(tmp_path / "backup.engram")
    with Engram(path=populated) as mem:
        mem.backup(dest)
    assert (tmp_path / "backup.engram").exists()


def test_backup_is_valid_sqlite(populated, tmp_path):
    dest = str(tmp_path / "backup.engram")
    with Engram(path=populated) as mem:
        mem.backup(dest)
    conn = sqlite3.connect(dest)
    row = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
    conn.close()
    assert row[0] == 2


def test_backup_includes_facts(populated, tmp_path):
    dest = str(tmp_path / "backup.engram")
    with Engram(path=populated) as mem:
        mem.backup(dest)
    conn = sqlite3.connect(dest)
    row = conn.execute("SELECT COUNT(*) FROM facts").fetchone()
    conn.close()
    assert row[0] == 1


def test_backup_openable_as_engram(populated, tmp_path):
    dest = str(tmp_path / "backup.engram")
    with Engram(path=populated) as mem:
        mem.backup(dest)
    with Engram(path=dest) as restored:
        assert restored._store.episode_count() == 2
        assert restored._store.fact_count() == 1


def test_backup_path_object(populated, tmp_path):
    dest = tmp_path / "backup.engram"
    with Engram(path=populated) as mem:
        mem.backup(dest)
    assert dest.exists()


def test_backup_live_store(populated, tmp_path):
    """Backup while the store is open and has unflushed writes."""
    dest = str(tmp_path / "live_backup.engram")
    with Engram(path=populated) as mem:
        mem.observe("Extra event added before backup")
        mem.backup(dest)
    conn = sqlite3.connect(dest)
    row = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
    conn.close()
    assert row[0] == 3


# ------------------------------------------------------------------
# backup() is a whole-file operation, so a scoped instance cannot have one
# ------------------------------------------------------------------


def test_backup_from_a_scoped_instance_refuses(tmp_path):
    """A file copy cannot be filtered, so an agent-scoped instance is refused
    rather than handed every other agent's raw episodes."""
    path = str(tmp_path / "team.engram")
    dest = tmp_path / "leak.engram"
    with Engram(path=path, agent_id="planner") as planner:
        planner.observe("planner drafted the Q3 roadmap")
    with Engram(path=path, agent_id="coder") as coder:
        coder.observe("coder wrote the retry loop")
        with pytest.raises(ValueError, match="whole file"):
            coder.backup(str(dest))
    assert not dest.exists()


def test_backup_refusal_names_the_alternatives(tmp_path):
    with Engram(path=":memory:", agent_id="coder") as coder, pytest.raises(ValueError) as excinfo:
        coder.backup(str(tmp_path / "leak.engram"))
    message = str(excinfo.value)
    assert "export_json" in message
    assert "agent_id" in message
