"""SQLite schema definition and migration runner."""

from __future__ import annotations

import contextlib
import sqlite3

import sqlite_vec  # type: ignore[import-untyped]

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
    importance_score REAL NOT NULL DEFAULT 1.0
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
    cost_tokens             INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS access_log (
    memory_id   TEXT NOT NULL,
    accessed_at DATETIME NOT NULL,
    query       TEXT,
    rank        INTEGER
);
"""


def migrate(conn: sqlite3.Connection, dim: int = DEFAULT_DIM) -> None:
    """Create all tables and the vector index. Idempotent — safe to call repeatedly."""
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript(_DDL)

    # Backfill columns for databases created before v0.2/v0.3.
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE episodes ADD COLUMN importance_score REAL NOT NULL DEFAULT 1.0")
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE facts ADD COLUMN extracted_by TEXT REFERENCES reflections(id)")

    # vec0 virtual table dimension is baked in at creation; IF NOT EXISTS guards re-runs.
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodes USING vec0(embedding float[{dim}])"
    )
    conn.commit()
