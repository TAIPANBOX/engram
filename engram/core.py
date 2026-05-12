"""Public Engram class — the main entry point for all memory operations."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engram.decay import run_decay
from engram.embedder import DEFAULT_MODEL, Embedder
from engram.importance import DecayConfig
from engram.models import Fact, ReflectionRun, SearchResult
from engram.retrieval import recall as _recall
from engram.schema import migrate
from engram.store import Store

if TYPE_CHECKING:
    from engram.llm import LLMAdapter


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
        llm: LLMAdapter | None = None,
    ) -> None:
        self._path = str(path)
        # check_same_thread=False: reflect_async() runs on a background thread
        # but SQLite serialises writes, so this is safe for our single-writer model.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._decay_cfg = decay_config or DecayConfig()
        self._llm = llm

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
    # Facts
    # ------------------------------------------------------------------

    def assert_fact(
        self,
        subject: str,
        predicate: str,
        object: str,  # noqa: A002
        *,
        confidence: float = 1.0,
        source: str | None = None,
    ) -> str:
        """Manually record a semantic fact triple without calling an LLM.

        Args:
            subject: Entity the fact is about (e.g. "Ivan").
            predicate: Relationship (e.g. "works_at").
            object: Value (e.g. "Globex").
            confidence: Certainty of the fact (0-1).
            source: Free-text provenance note (stored in derived_from).

        Returns:
            The fact id (UUID string).
        """
        now = datetime.now(tz=UTC)
        fact_id = str(uuid.uuid4())
        fact = Fact(
            id=fact_id,
            subject=subject,
            predicate=predicate,
            object=object,
            valid_from=now,
            valid_to=None,
            recorded_at=now,
            superseded_at=None,
            superseded_by=None,
            confidence=confidence,
            derived_from=[source] if source else [],
            extracted_by=None,
        )
        self._store.insert_fact(fact)
        return fact_id

    def why(self, fact_id: str) -> dict[str, Any]:
        """Return provenance information for a fact.

        Args:
            fact_id: Id of the fact to explain.

        Returns:
            Dict with keys ``fact``, ``extracted_from``, ``extracted_by``,
            ``confidence``, ``model``.

        Raises:
            KeyError: If the fact_id is not found.
        """
        fact = self._store.get_fact(fact_id)
        if fact is None:
            raise KeyError(f"Fact not found: {fact_id!r}")
        model_used: str | None = None
        if fact.extracted_by:
            run = self._store.get_reflection_by_id(fact.extracted_by)
            if run:
                model_used = run.model_used
        return {
            "fact": f"{fact.subject} {fact.predicate} {fact.object}",
            "extracted_from": fact.derived_from,
            "extracted_by": fact.extracted_by,
            "confidence": fact.confidence,
            "model": model_used,
        }

    def contradictions(self) -> list[tuple[Fact, Fact]]:
        """Return pairs of active facts that share (subject, predicate) but differ in object.

        Returns:
            List of (fact_a, fact_b) pairs, each pair representing a conflict.
        """
        all_active = self._store.get_all_active_facts()
        groups: dict[tuple[str, str], list[Fact]] = defaultdict(list)
        for fact in all_active:
            groups[(fact.subject, fact.predicate)].append(fact)
        pairs: list[tuple[Fact, Fact]] = []
        for facts in groups.values():
            if len(facts) > 1:
                for i in range(len(facts)):
                    for j in range(i + 1, len(facts)):
                        pairs.append((facts[i], facts[j]))
        return pairs

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------

    def reflect(self) -> ReflectionRun:
        """Run the reflection loop synchronously.

        Returns:
            The completed :class:`ReflectionRun`.
        """
        from engram.reflection import reflect as _reflect

        return _reflect(self._store, self._llm, self._decay_cfg)

    def reflect_async(self) -> threading.Thread:
        """Run the reflection loop in a background thread.

        Returns:
            The started :class:`threading.Thread`. Call ``.join()`` to wait for completion.
        """
        thread = threading.Thread(target=self.reflect, daemon=True)
        thread.start()
        return thread

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
