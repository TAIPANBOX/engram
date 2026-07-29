"""SQLite schema definition and migration runner."""

from __future__ import annotations

import contextlib
import re
import sqlite3

import sqlite_vec  # type: ignore[import-untyped]

# Collect all OperationalError types that may appear depending on which
# sqlite backend is in use (sqlite3 vs sqlcipher3.dbapi2).
_OP_ERRORS: tuple[type[BaseException], ...] = (sqlite3.OperationalError,)
try:
    import sqlcipher3.dbapi2 as _sc

    _OP_ERRORS = (*_OP_ERRORS, _sc.OperationalError)
except ImportError:
    pass

# Embedding dimension for bge-small-en-v1.5
DEFAULT_DIM: int = 384

# vec0 preallocates one chunk per partition, so the value trades scan speed
# against wasted space in stores with many small agents. Measured at 50k
# vectors / 384 dim: chunk_size=128 scans ~4% slower than the 1024 default,
# while 50 single-episode agents cost 10 MB instead of 80 MB.
_VEC_CHUNK_SIZE = 128


def _vec_ddl(dim: int) -> str:
    """DDL for the vector index.

    ``agent_id`` is a partition key and ``ts`` a metadata column so that
    scoped and ``as_of`` recall are resolved *inside* the KNN scan. Both used
    to be applied in an outer JOIN, which could only ever cut into an already
    chosen global top-k: an agent whose episodes were not globally nearest got
    an empty result instead of its own nearest ones.
    """
    return (
        "CREATE VIRTUAL TABLE vec_episodes USING vec0("
        "agent_id text partition key, "
        "ts text, "
        f"embedding float[{dim}], "
        f"chunk_size={_VEC_CHUNK_SIZE})"
    )


_DDL = """
CREATE TABLE IF NOT EXISTS episodes (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    timestamp       DATETIME NOT NULL,
    actors          JSON DEFAULT '[]',
    tags            JSON DEFAULT '[]',
    salience        REAL DEFAULT 0.5,
    emotional_valence REAL DEFAULT 0.0,
    summary_of      JSON DEFAULT '[]',
    importance_score REAL NOT NULL DEFAULT 1.0,
    agent_id        TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id              TEXT PRIMARY KEY,
    subject         TEXT NOT NULL,
    predicate       TEXT NOT NULL,
    object          TEXT NOT NULL,
    valid_from      DATETIME NOT NULL,
    valid_to        DATETIME,
    recorded_at     DATETIME NOT NULL,
    superseded_at   DATETIME,
    superseded_by   TEXT REFERENCES facts(id),
    confidence      REAL NOT NULL DEFAULT 1.0,
    derived_from    JSON DEFAULT '[]',
    extracted_by    TEXT REFERENCES reflections(id)
);

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,
    aliases     JSON DEFAULT '[]',
    first_seen  DATETIME,
    last_seen   DATETIME
);

CREATE TABLE IF NOT EXISTS edges (
    src_id      TEXT NOT NULL,
    dst_id      TEXT NOT NULL,
    relation    TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    created_at  DATETIME NOT NULL,
    agent_id    TEXT DEFAULT NULL,
    PRIMARY KEY (src_id, dst_id, relation)
);

CREATE TABLE IF NOT EXISTS reflections (
    id                      TEXT PRIMARY KEY,
    started_at              DATETIME NOT NULL,
    finished_at             DATETIME,
    episodes_processed      INTEGER DEFAULT 0,
    facts_extracted         INTEGER DEFAULT 0,
    contradictions_resolved INTEGER DEFAULT 0,
    model_used              TEXT,
    cost_tokens             INTEGER DEFAULT 0,
    agent_id                TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS access_log (
    memory_id   TEXT NOT NULL,
    accessed_at DATETIME NOT NULL,
    query       TEXT,
    rank        INTEGER,
    agent_id    TEXT DEFAULT NULL
);
"""


def _repartition_vec_episodes(conn: sqlite3.Connection, dim: int) -> None:
    """Rebuild a pre-v2.3 flat vec0 table as a partitioned one.

    A vec0 table's partition keys and metadata columns are fixed at creation,
    so the only way forward for an existing store is to copy the vectors out,
    drop the table, and re-insert them alongside the ``agent_id`` and
    ``timestamp`` their episodes already carry. Runs once per store, costs one
    O(N) pass, and holds every vector in memory while it does: the vectors have
    to be read before the table they live in can be dropped.

    Orphan vectors (no surviving episode row) keep a NULL partition, exactly
    where an unscoped write would have put them, and an empty ``ts``: vec0
    accepts NULL in a partition key but rejects it in a TEXT metadata column.
    The empty string never reaches a result, because every search path inner
    joins ``episodes`` and an orphan has no row there to join to.
    """
    rows = conn.execute(
        "SELECT v.rowid, e.agent_id, COALESCE(e.timestamp, ''), v.embedding "
        "FROM vec_episodes v LEFT JOIN episodes e ON e.rowid = v.rowid"
    ).fetchall()
    conn.execute("DROP TABLE vec_episodes")
    conn.execute(_vec_ddl(dim))
    conn.executemany(
        "INSERT INTO vec_episodes(rowid, agent_id, ts, embedding) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def migrate(conn: sqlite3.Connection, dim: int = DEFAULT_DIM) -> None:
    """Create all tables and the vector index. Idempotent — safe to call repeatedly."""
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript(_DDL)

    # Backfill columns for databases created before v0.2/v0.3.
    with contextlib.suppress(*_OP_ERRORS):
        conn.execute("ALTER TABLE episodes ADD COLUMN importance_score REAL NOT NULL DEFAULT 1.0")
    with contextlib.suppress(*_OP_ERRORS):
        conn.execute("ALTER TABLE facts ADD COLUMN extracted_by TEXT REFERENCES reflections(id)")

    # Backfill columns for databases created before v1.3 (multi-agent).
    with contextlib.suppress(*_OP_ERRORS):
        conn.execute("ALTER TABLE episodes ADD COLUMN agent_id TEXT DEFAULT NULL")
    with contextlib.suppress(*_OP_ERRORS):
        conn.execute("ALTER TABLE reflections ADD COLUMN agent_id TEXT DEFAULT NULL")
    with contextlib.suppress(*_OP_ERRORS):
        conn.execute("ALTER TABLE access_log ADD COLUMN agent_id TEXT DEFAULT NULL")

    # Backfill the edge agent scope (added in v2.2): episode->entity edges are
    # private to the agent that created them, so spreading activation stays
    # within an agent while entities/facts remain shared.
    with contextlib.suppress(*_OP_ERRORS):
        conn.execute("ALTER TABLE edges ADD COLUMN agent_id TEXT DEFAULT NULL")

    with contextlib.suppress(*_OP_ERRORS):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(agent_id)")
    with contextlib.suppress(*_OP_ERRORS):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_agent ON edges(agent_id)")

    # The vec0 dimension and layout are baked in at creation, so an existing
    # table is inspected rather than recreated.
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'vec_episodes'").fetchone()
    existing = row[0] if row is not None and row[0] else None

    if existing is None:
        conn.execute(_vec_ddl(dim))
    else:
        # Guard against re-opening an existing store with a different embedder:
        # the vec0 dimension is immutable, so a mismatch would silently corrupt
        # search (inserts/queries with wrong-size vectors). Fail loudly instead.
        match = re.search(r"float\[(\d+)\]", existing)
        if match is not None and int(match.group(1)) != dim:
            raise ValueError(
                f"embedding dimension mismatch: this store was created with "
                f"dim={match.group(1)}, but the current embedder produces dim={dim}. "
                f"Open the store with the original embedder model, or export and "
                f"re-import to re-embed with the new model."
            )
        if "partition key" not in existing:
            _repartition_vec_episodes(conn, dim)

    # FTS5 full-text index over episode content (added in v2.1).
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_episodes "
        "USING fts5(content, content='episodes', content_rowid='rowid')"
    )
    # Populate FTS for any pre-existing episodes that predate this migration.
    conn.execute(
        "INSERT INTO fts_episodes(rowid, content) "
        "SELECT rowid, content FROM episodes "
        "WHERE rowid NOT IN (SELECT rowid FROM fts_episodes)"
    )
    conn.commit()
