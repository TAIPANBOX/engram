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
