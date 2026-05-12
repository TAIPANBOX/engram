"""Low-level CRUD operations over the SQLite schema."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

import numpy as np

from engram.models import EmbeddedVector, Episode


def _serialize(vec: EmbeddedVector) -> bytes:
    return vec.astype(np.float32).tobytes()


def _distance_to_score(distance: float) -> float:
    """Convert L2 distance from sqlite-vec to a [0, 1] cosine-like score."""
    return float(1.0 / (1.0 + distance))


class Store:
    """Thin wrapper around a sqlite3 connection providing episode persistence."""

    def __init__(self, conn: sqlite3.Connection, dim: int) -> None:
        self._conn = conn
        self._dim = dim

    def insert_episode(self, ep: Episode, embedding: EmbeddedVector) -> None:
        """Persist an episode and its embedding vector."""
        cur = self._conn.execute(
            """
            INSERT INTO episodes (id, content, timestamp, actors, tags,
                                  salience, emotional_valence, summary_of)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ep.id,
                ep.content,
                ep.timestamp.isoformat(),
                json.dumps(ep.actors),
                json.dumps(ep.tags),
                ep.salience,
                ep.emotional_valence,
                json.dumps(ep.summary_of),
            ),
        )
        rowid: int = cur.lastrowid  # type: ignore[assignment]
        self._conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
            (rowid, _serialize(embedding)),
        )
        self._conn.commit()

    def get_episode(self, episode_id: str) -> Episode | None:
        """Fetch a single episode by id."""
        row: Any = self._conn.execute(
            "SELECT id, content, timestamp, actors, tags, salience, emotional_valence, "
            "summary_of, importance_score FROM episodes WHERE id = ?",
            (episode_id,),
        ).fetchone()
        return Episode.from_row(tuple(row)) if row else None

    def search_episodes(
        self, query_embedding: EmbeddedVector, k: int
    ) -> list[tuple[Episode, float, float]]:
        """Return top-k episodes by vector similarity.

        Returns list of (episode, score, distance) triples, best first.
        """
        rows: list[Any] = self._conn.execute(
            """
            SELECT e.id, e.content, e.timestamp, e.actors, e.tags,
                   e.salience, e.emotional_valence, e.summary_of, e.importance_score,
                   v.distance
            FROM vec_episodes v
            JOIN episodes e ON e.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance ASC
            """,
            (_serialize(query_embedding), k),
        ).fetchall()

        results: list[tuple[Episode, float, float]] = []
        for row in rows:
            ep = Episode.from_row(tuple(row[:9]))
            distance = float(row[9])
            score = _distance_to_score(distance)
            results.append((ep, score, distance))
        return results

    def log_access(
        self, memory_id: str, accessed_at: datetime, query: str | None, rank: int
    ) -> None:
        """Record a retrieval event in the access log."""
        self._conn.execute(
            "INSERT INTO access_log (memory_id, accessed_at, query, rank) VALUES (?, ?, ?, ?)",
            (memory_id, accessed_at.isoformat(), query, rank),
        )
        self._conn.commit()

    def get_access_stats(self, memory_id: str) -> tuple[int, datetime | None]:
        """Return (access_count, last_accessed_at) for a memory."""
        row: Any = self._conn.execute(
            "SELECT COUNT(*), MAX(accessed_at) FROM access_log WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        count = int(row[0]) if row[0] else 0
        last: datetime | None = None
        if row[1]:
            last = datetime.fromisoformat(row[1])
        return count, last

    def update_importance(self, memory_id: str, score: float) -> None:
        """Persist a newly computed importance score."""
        self._conn.execute(
            "UPDATE episodes SET importance_score = ? WHERE id = ?",
            (score, memory_id),
        )
        self._conn.commit()

    def get_episodes_for_decay(self) -> list[tuple[str, float, float, datetime]]:
        """Return minimal episode data needed by the decay job.

        Returns list of (id, salience, emotional_valence, timestamp).
        """
        rows: list[Any] = self._conn.execute(
            "SELECT id, salience, emotional_valence, timestamp FROM episodes"
        ).fetchall()
        return [
            (row[0], float(row[1]), float(row[2]), datetime.fromisoformat(row[3]))
            for row in rows
        ]

    def episode_count(self) -> int:
        row: Any = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return int(row[0])

    def vec_count(self) -> int:
        row: Any = self._conn.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()
        return int(row[0])
