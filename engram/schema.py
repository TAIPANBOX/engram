"""SQLite schema definition and migration runner."""

from __future__ import annotations

import contextlib
import sqlite3

import sqlite_vec  # type: ignore[import-untyped]

# Collect all OperationalError types that may appear depending on which
# sqlite backend is in use (sqlite3 vs sqlcipher3.dbapi2).
_OP_ERRORS: tuple[type[BaseException], ...] = (sqlite3.OperationalError,)
try:
    import sqlcipher3.dbapi2 as _sc  # type: ignore[import-untyped]

    _OP_ERRORS = (*_OP_ERRORS, _sc.OperationalError)
except ImportError:
    pass

# Embedding dimension for bge-small-en-v1.5
DEFAULT_DIM: int = 384

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

    with contextlib.suppress(*_OP_ERRORS):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(agent_id)")

    # vec0 virtual table dimension is baked in at creation; IF NOT EXISTS guards re-runs.
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodes USING vec0(embedding float[{dim}])"
    )

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
