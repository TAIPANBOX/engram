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


def test_interrupted_repartition_keeps_the_vectors(tmp_path, monkeypatch) -> None:
    """An interrupted migration must roll back to the flat table rather than
    leave episodes whose embeddings are gone and cannot be recomputed."""
    from engram import Engram
    from engram.schema import _vec_ddl

    path = str(tmp_path / "interrupted.engram")
    with Engram(path=path, agent_id="a") as mem:
        for i in range(10):
            mem.observe(f"Episode number {i}")

    _downgrade_to_flat_vec(path)

    boom = RuntimeError("process died mid-migration")

    def explode(dim: int) -> str:
        _vec_ddl(dim)
        raise boom

    monkeypatch.setattr("engram.schema._vec_ddl", explode)
    with pytest.raises(RuntimeError):
        Engram(path=path, agent_id="a")
    monkeypatch.undo()

    # The flat table is back, with every vector still in it.
    import sqlite_vec

    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'vec_episodes'").fetchone()[0]
    assert "partition key" not in ddl
    assert conn.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()[0] == 10
    conn.close()

    # And the next open migrates cleanly.
    with Engram(path=path, agent_id="a") as mem:
        assert mem._store.vec_count() == 10
        assert len(mem.recall("Episode", k=5)) == 5


def _downgrade_to_pre_fts(path: str) -> None:
    """Strip a store back to how it looked before v2.1 added the FTS index.

    Episodes and their vectors stay; the full-text index goes, along with the
    marker that says the index has been built. The current code cannot write
    such a file, so this is the only way to exercise the migration over rows
    that were already there.
    """
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE fts_episodes")
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()


def test_migrate_indexes_episodes_that_predate_the_fts_table(tmp_path) -> None:
    """Every episode written before v2.1 was missing from the FTS index.

    The backfill filtered on ``rowid NOT IN (SELECT rowid FROM fts_episodes)``.
    ``fts_episodes`` is an external-content table, so that scan reads through to
    ``episodes`` and returns every rowid, indexed or not, and the predicate
    matched nothing on any store that has ever run it.
    """
    from engram import Engram

    path = str(tmp_path / "pre_fts.engram")
    with Engram(path=path) as mem:
        mem.observe("The restaurant I mentioned earlier was Osteria Bianca.")
        mem.observe("The migration finished on Tuesday and nothing broke.")

    _downgrade_to_pre_fts(path)

    with Engram(path=path) as mem:
        rows = mem._conn.execute(
            "SELECT rowid FROM fts_episodes WHERE fts_episodes MATCH 'Osteria'"
        ).fetchall()
        assert rows, "an episode written before v2.1 never reached the FTS index"


def test_pre_fts_episodes_are_ranked_by_bm25_after_migration(tmp_path) -> None:
    """The user-visible half of the same defect, and the reason it is not cosmetic.

    Since 2.3.0 made hybrid the default, an episode missing from the index
    scores 0.0 on the BM25 half of every recall and sits below anything written
    after the upgrade. The store is degraded rather than empty, so it fails
    silently.
    """
    from engram import Engram

    path = str(tmp_path / "pre_fts_rank.engram")
    with Engram(path=path) as mem:
        mem.observe("The restaurant I mentioned earlier was Osteria Bianca.")
        mem.observe("The migration finished on Tuesday and nothing broke.")

    _downgrade_to_pre_fts(path)

    with Engram(path=path) as mem:
        results = mem.recall(
            "which restaurant was mentioned", k=2, mode="hybrid", vector_weight=0.0, fts_weight=1.0
        )
        assert results, "the BM25 half of hybrid recall returned nothing at all"
        assert results[0].score > 0.0, "every score was 0.0, so BM25 ranked none of them"
        assert "Osteria Bianca" in results[0].episode.content

        # And the migration did not cost the insert-time population: an episode
        # written into the migrated store is indexed the moment it lands.
        mem.observe("Osteria Bianca took the booking for Friday.")
        fresh = mem.recall(
            "booking on Friday", k=1, mode="hybrid", vector_weight=0.0, fts_weight=1.0
        )
        assert fresh and "Friday" in fresh[0].episode.content
        assert fresh[0].score > 0.0


def test_an_unqualified_scan_of_the_fts_table_reads_through_to_episodes(
    conn: sqlite3.Connection,
) -> None:
    """Pins the trap itself, because it looks like a working predicate.

    Counting or listing an external-content FTS5 table tells you about
    ``episodes``, not about the index. Any future attempt to decide "is this
    row indexed?" that way will be wrong in exactly the same silent direction,
    so the behaviour is asserted rather than described in a comment.
    """
    migrate(conn)
    conn.execute(
        "INSERT INTO episodes (id, content, timestamp) VALUES ('e1', 'sphinx quartz', '2026-01-01')"
    )
    conn.commit()

    assert conn.execute("SELECT rowid FROM fts_episodes").fetchall() == [(1,)]
    assert conn.execute("SELECT count(*) FROM fts_episodes").fetchone()[0] == 1
    assert (
        conn.execute(
            "SELECT rowid FROM episodes WHERE rowid NOT IN (SELECT rowid FROM fts_episodes)"
        ).fetchall()
        == []
    ), "the old backfill's predicate matches nothing, on every store"

    # All of which says the row is indexed. It is not.
    assert (
        conn.execute("SELECT rowid FROM fts_episodes WHERE fts_episodes MATCH 'sphinx'").fetchall()
        == []
    )


def test_the_fts_rebuild_is_gated_on_the_schema_marker(conn: sqlite3.Connection) -> None:
    """The rebuild is O(N) in the whole store, so it runs once, not per open.

    The measurement behind that choice is in `_rebuild_fts_once`; this pins the
    behaviour. Both directions are asserted, because a marker that is never
    read reintroduces the cost and a marker that is never written reintroduces
    the bug, and each failure is invisible from the other side.
    """
    from engram.schema import _SCHEMA_VERSION

    migrate(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION

    # A row written straight into episodes, which is what a pre-v2.1 writer did.
    conn.execute(
        "INSERT INTO episodes (id, content, timestamp) VALUES ('e1', 'sphinx quartz', '2026-01-01')"
    )
    conn.commit()

    def matched() -> list[object]:
        return conn.execute(
            "SELECT rowid FROM fts_episodes WHERE fts_episodes MATCH 'sphinx'"
        ).fetchall()

    migrate(conn)
    assert matched() == [], "the marker was stamped and the rebuild ran anyway"

    conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION - 1}")
    migrate(conn)
    assert matched() == [(1,)], "the marker was cleared and the rebuild did not run"


def test_repartition_warns_about_the_format_floor(tmp_path) -> None:
    """The rewrite is irreversible for older sqlite-vec installs, so it must
    not happen silently."""
    from engram import Engram

    path = str(tmp_path / "warns.engram")
    with Engram(path=path, agent_id="a") as mem:
        mem.observe("Something worth keeping")

    _downgrade_to_flat_vec(path)

    with pytest.warns(UserWarning, match="sqlite-vec >= 0.1.6"):
        Engram(path=path, agent_id="a").close()

    # Second open is already migrated, so it stays quiet.
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        Engram(path=path, agent_id="a").close()
