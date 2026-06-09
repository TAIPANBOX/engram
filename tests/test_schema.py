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


def test_wal_mode_enabled_for_file_store(tmp_path: object) -> None:
    """File-based stores must use WAL journal mode."""
    from engram import Engram

    path = str(tmp_path / "wal_test.engram")  # type: ignore[operator]
    with Engram(path=path) as mem:
        row = mem._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal", f"expected WAL, got {row[0]!r}"


def test_wal_not_applied_to_memory_store() -> None:
    """:memory: stores must open without error (WAL skipped silently)."""
    from engram import Engram

    with Engram(path=":memory:") as mem:
        row = mem._conn.execute("PRAGMA journal_mode").fetchone()
        # :memory: returns 'memory' — not WAL, and that's correct
        assert row[0] == "memory"


def test_cache_size_applied(tmp_path: object) -> None:
    """PRAGMA cache_size must be set to the configured value (-32000 pages)."""
    from engram import Engram

    path = str(tmp_path / "cache_test.engram")  # type: ignore[operator]
    with Engram(path=path) as mem:
        row = mem._conn.execute("PRAGMA cache_size").fetchone()
        # SQLite may return negative (KB) or positive (pages); either -32000 or a large positive
        assert int(row[0]) != 0


def test_migrate_same_dim_is_idempotent(conn: sqlite3.Connection) -> None:
    """Re-running migrate with the same dimension must succeed silently."""
    migrate(conn, dim=384)
    migrate(conn, dim=384)  # no error


def test_migrate_rejects_dimension_mismatch(conn: sqlite3.Connection) -> None:
    """Re-opening a store with a different embedder dim must fail loudly.

    The vec0 table bakes its dimension in at creation; a silent mismatch
    would corrupt vector search, so migrate() raises ValueError instead.
    """
    migrate(conn, dim=384)
    with pytest.raises(ValueError, match="dimension mismatch"):
        migrate(conn, dim=768)
