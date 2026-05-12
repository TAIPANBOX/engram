"""Async wrapper around Engram for use in asyncio-based agents.

All methods delegate to the synchronous :class:`~engram.core.Engram` instance
via :func:`asyncio.get_event_loop().run_in_executor`, keeping the event loop
unblocked during ONNX inference and SQLite I/O.

Example::

    async with AsyncEngram(path="memory.engram") as mem:
        await mem.observe("Alice joined the project")
        results = await mem.recall("who joined the project?")
        for r in results:
            print(r.episode.content)
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from engram.core import Engram
from engram.importance import DecayConfig
from engram.models import Fact, ForgetResult, ObserveInput, SearchResult

try:
    from engram.llm import LLMAdapter
except ImportError:
    LLMAdapter = Any  # type: ignore[assignment,misc]


class AsyncEngram:
    """Async-compatible Engram store.

    Wraps a synchronous :class:`~engram.core.Engram` instance; every I/O
    method is an ``async def`` that executes on the default thread-pool
    executor so the event loop is never blocked.

    Args:
        path: Path to the ``.engram`` SQLite file, or ``":memory:"``.
        embedder_model: fastembed model name.
        decay_config: Importance decay parameters.
        llm: LLM adapter for reflection (optional).
        agent_id: Scope this instance to a named agent.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        embedder_model: str | None = None,
        decay_config: DecayConfig | None = None,
        llm: Any | None = None,
        agent_id: str | None = None,
        key: str | None = None,
    ) -> None:
        from engram.embedder import DEFAULT_MODEL

        self._engram = Engram(
            path=path,
            embedder_model=embedder_model or DEFAULT_MODEL,
            decay_config=decay_config,
            llm=llm,
            agent_id=agent_id,
            key=key,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> AsyncEngram:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Flush and close the underlying store."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._engram.close)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def observe(
        self,
        content: str,
        *,
        actors: list[str] | None = None,
        tags: list[str] | None = None,
        salience: float = 0.5,
        emotional_valence: float = 0.0,
    ) -> str:
        """Async version of :meth:`~engram.core.Engram.observe`.

        Returns:
            Episode id (UUID string).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._engram.observe(
                content,
                actors=actors,
                tags=tags,
                salience=salience,
                emotional_valence=emotional_valence,
            ),
        )

    async def observe_many(self, items: list[ObserveInput]) -> list[str]:
        """Async version of :meth:`~engram.core.Engram.observe_many`.

        Returns:
            Episode ids in the same order as *items*.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._engram.observe_many(items))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def recall(
        self,
        query: str,
        k: int = 5,
        *,
        mode: str = "cosine",
        depth: int = 2,
        decay: float = 0.5,
        vector_weight: float = 0.7,
        fts_weight: float = 0.3,
        as_of: datetime | None = None,
        cross_agent: bool = False,
    ) -> list[SearchResult]:
        """Async version of :meth:`~engram.core.Engram.recall`.

        Returns:
            List of :class:`~engram.models.SearchResult` ordered by score descending.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._engram.recall(
                query,
                k,
                mode=mode,
                depth=depth,
                decay=decay,
                vector_weight=vector_weight,
                fts_weight=fts_weight,
                as_of=as_of,
                cross_agent=cross_agent,
            ),
        )

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    async def assert_fact(
        self,
        subject: str,
        predicate: str,
        object: str,  # noqa: A002
        *,
        confidence: float = 1.0,
        source: str | None = None,
    ) -> str:
        """Async version of :meth:`~engram.core.Engram.assert_fact`.

        Returns:
            Fact id (UUID string).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._engram.assert_fact(
                subject, predicate, object, confidence=confidence, source=source
            ),
        )

    async def timeline(self, entity: str) -> list[Fact]:
        """Async version of :meth:`~engram.core.Engram.timeline`."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._engram.timeline(entity))

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def decay(self) -> int:
        """Async version of :meth:`~engram.core.Engram.decay`.

        Returns:
            Number of episodes updated.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._engram.decay)

    async def backup(self, dest: str | Path) -> None:
        """Async version of :meth:`~engram.core.Engram.backup`."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._engram.backup(dest))

    async def export_json(self, dest: str) -> dict[str, Any]:
        """Async version of :meth:`~engram.core.Engram.export_json`."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._engram.export_json(dest))

    async def import_json(self, src: str, *, merge: bool = False) -> dict[str, int]:
        """Async version of :meth:`~engram.core.Engram.import_json`."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._engram.import_json(src, merge=merge))

    # ------------------------------------------------------------------
    # Forget
    # ------------------------------------------------------------------

    async def forget(self, episode_id: str, *, reason: str | None = None) -> None:
        """Async version of :meth:`~engram.core.Engram.forget`."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._engram.forget(episode_id, reason=reason))

    async def forget_entity(self, entity_name: str) -> ForgetResult:
        """Async version of :meth:`~engram.core.Engram.forget_entity`."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._engram.forget_entity(entity_name))

    # ------------------------------------------------------------------
    # Pass-through properties
    # ------------------------------------------------------------------

    @property
    def _store(self) -> Any:
        """Direct access to the underlying store (for testing)."""
        return self._engram._store
