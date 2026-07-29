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


# ------------------------------------------------------------------
# Repartitioning a pre-v2.3 store
# ------------------------------------------------------------------


def _downgrade_to_flat_vec(path: str, dim: int = 384) -> None:
    """Rewrite a store's vector index in the pre-v2.3 flat layout.

    Reproduces a file written before agent_id became a partition key, which is
    the only way to exercise the migration: the current code cannot create one.
    """
    import sqlite_vec

    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    rows = conn.execute("SELECT rowid, embedding FROM vec_episodes").fetchall()
    conn.execute("DROP TABLE vec_episodes")
    conn.execute(f"CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding float[{dim}])")
    conn.executemany("INSERT INTO vec_episodes(rowid, embedding) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


def test_migrate_repartitions_a_flat_store(tmp_path) -> None:
    from engram import Engram

    path = str(tmp_path / "old.engram")
    with Engram(path=path, agent_id="noisy") as noisy:
        for i in range(40):
            noisy.observe(f"Deployment log line {i} from the primary cluster")
    with Engram(path=path, agent_id="quiet") as quiet:
        for i in range(3):
            quiet.observe(f"Deployment rollback note {i} from the primary cluster")

    _downgrade_to_flat_vec(path)

    ddl = (
        sqlite3.connect(path)
        .execute("SELECT sql FROM sqlite_master WHERE name = 'vec_episodes'")
        .fetchone()[0]
    )
    assert "partition key" not in ddl

    # Opening the store runs migrate(), which must rebuild the index in place.
    with Engram(path=path, agent_id="quiet") as quiet:
        ddl = quiet._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'vec_episodes'"
        ).fetchone()[0]
        assert "partition key" in ddl

        # Vectors kept their rowids and gained the right partition, so the
        # scoped query the old layout answered with nothing now works.
        results = quiet.recall("deployment cluster", k=3)
        assert len(results) == 3
        assert all("rollback note" in r.episode.content for r in results)

    with Engram(path=path) as everyone:
        assert everyone._store.vec_count() == 43
        assert len(everyone.recall("deployment cluster", k=43)) == 43


def test_repartition_preserves_orphan_vectors(tmp_path) -> None:
    """A vector whose episode row is gone has no agent to inherit, so it keeps
    the NULL partition rather than blocking the migration."""
    from engram import Engram

    path = str(tmp_path / "orphan.engram")
    with Engram(path=path, agent_id="a") as mem:
        mem.observe("Episode that will lose its row")
        mem.observe("Episode that survives")
        mem._conn.execute("DELETE FROM episodes WHERE content LIKE 'Episode that will lose%'")
        mem._conn.commit()

    _downgrade_to_flat_vec(path)

    with Engram(path=path, agent_id="a") as mem:
        assert mem._store.vec_count() == 2
        assert len(mem.recall("survives", k=5)) == 1
