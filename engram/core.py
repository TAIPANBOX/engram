"""Public Engram class — the main entry point for all memory operations."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from engram.decay import run_decay
from engram.embedder import DEFAULT_MODEL, Embedder
from engram.importance import DecayConfig
from engram.models import SearchResult
from engram.retrieval import recall as _recall
from engram.schema import migrate
from engram.store import Store


class Engram:
    """Single-file cognitive memory store for AI agents.

    Args:
        path: Path to the .engram SQLite file, or ``":memory:"`` for an in-process store.
        embedder_model: fastembed model name used for all embeddings.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        embedder_model: str = DEFAULT_MODEL,
        decay_config: DecayConfig | None = None,
    ) -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._decay_cfg = decay_config or DecayConfig()

        self._embedder = Embedder(embedder_model)
        migrate(self._conn, dim=self._embedder.dim)
        self._store = Store(self._conn, dim=self._embedder.dim)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def observe(
        self,
        content: str,
        *,
        actors: list[str] | None = None,
        tags: list[str] | None = None,
        salience: float = 0.5,
        emotional_valence: float = 0.0,
    ) -> str:
        """Record a new episodic observation.

        Args:
            content: Raw text of the observed event.
            actors: Named entities involved in the event.
            tags: Categorical labels for filtering.
            salience: Subjective importance at encoding time (0-1).
            emotional_valence: Affective weight (-1 negative to +1 positive).

        Returns:
            The episode id (UUID string).
        """
        from engram.models import Episode

        episode_id = str(uuid.uuid4())
        ep = Episode(
            id=episode_id,
            content=content,
            timestamp=datetime.now(tz=UTC),
            actors=actors or [],
            tags=tags or [],
            salience=salience,
            emotional_valence=emotional_valence,
        )
        embedding = self._embedder.embed(content)
        self._store.insert_episode(ep, embedding)
        return episode_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def recall(self, query: str, k: int = 5) -> list[SearchResult]:
        """Retrieve the top-k episodes most semantically similar to *query*.

        Args:
            query: Natural-language search query.
            k: Maximum number of results to return.

        Returns:
            List of :class:`SearchResult` ordered by descending similarity.
        """
        return _recall(query, k, self._store, self._embedder)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def decay(self) -> int:
        """Recompute importance scores for all episodes (synchronous).

        Returns:
            Number of episodes updated.
        """
        return run_decay(self._store, self._decay_cfg)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> Engram:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
