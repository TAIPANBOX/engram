"""Tests for schema creation and migration idempotency."""

import sqlite3

import pytest

from engram.schema import migrate


@pytest.fixture()
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows}


def test_migrate_creates_all_tables(conn: sqlite3.Connection) -> None:
    migrate(conn)
    tables = _table_names(conn)
    for expected in ("episodes", "facts", "entities", "edges", "reflections", "access_log"):
        assert expected in tables, f"missing table: {expected}"


def test_migrate_creates_vec_episodes(conn: sqlite3.Connection) -> None:
    migrate(conn)
    # vec0 virtual tables appear in sqlite_master as type='table'
    all_names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master").fetchall()}
    assert "vec_episodes" in all_names


def test_migrate_is_idempotent(conn: sqlite3.Connection) -> None:
    migrate(conn)
    migrate(conn)  # must not raise
    assert len(_table_names(conn)) >= 6
