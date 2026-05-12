"""Low-level CRUD operations over the SQLite schema."""

from __future__ import annotations

import json
import sqlite3
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
            "SELECT id, content, timestamp, actors, tags, salience, emotional_valence, summary_of "
            "FROM episodes WHERE id = ?",
            (episode_id,),
        ).fetchone()
        return Episode.from_row(row) if row else None

    def search_episodes(
        self, query_embedding: EmbeddedVector, k: int
    ) -> list[tuple[Episode, float, float]]:
        """Return top-k episodes by vector similarity.

        Returns list of (episode, score, distance) triples, best first.
        """
        rows: list[Any] = self._conn.execute(
            """
            SELECT e.id, e.content, e.timestamp, e.actors, e.tags,
                   e.salience, e.emotional_valence, e.summary_of,
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
            ep = Episode.from_row(row[:8])
            distance = float(row[8])
            score = _distance_to_score(distance)
            results.append((ep, score, distance))
        return results

    def episode_count(self) -> int:
        row: Any = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return int(row[0])

    def vec_count(self) -> int:
        row: Any = self._conn.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()
        return int(row[0])
