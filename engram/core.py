"""Public Engram class — the main entry point for all memory operations."""

from __future__ import annotations

import importlib
import sqlite3
import threading
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from engram.decay import run_decay
from engram.embedder import DEFAULT_MODEL, Embedder
from engram.events import EventLog, resolve_events_path
from engram.importance import DecayConfig
from engram.models import (
    CompressionRun,
    Fact,
    ForgetResult,
    ObserveInput,
    ReflectionRun,
    SearchResult,
)
from engram.retrieval import recall as _recall
from engram.schema import migrate
from engram.store import Store

if TYPE_CHECKING:
    from engram.llm import LLMAdapter


class ReflectionThread(threading.Thread):
    """Background thread returned by :meth:`Engram.reflect_async`.

    After :meth:`join` returns, access the completed run via :attr:`result`.
    Any exception raised inside the thread is re-raised on ``join()``.

    Example::

        thread = mem.reflect_async()
        thread.join()
        print(thread.result.facts_extracted)
    """

    def __init__(self, engram: Engram) -> None:
        super().__init__(daemon=True)
        self._engram = engram
        self.result: ReflectionRun | None = None
        self._exc: BaseException | None = None

    def run(self) -> None:
        try:
            self.result = self._engram.reflect()
        except BaseException as exc:
            self._exc = exc

    def join(self, timeout: float | None = None) -> None:
        super().join(timeout)
        if self._exc is not None:
            raise self._exc


def _sqlite_module(key: str | None) -> Any:
    """Return the sqlite3 module, or sqlcipher3.dbapi2 when a key is provided."""
    if key is None:
        return sqlite3
    try:
        return importlib.import_module("sqlcipher3.dbapi2")
    except ImportError as exc:
        raise ImportError(
            "Encryption requires the 'sqlcipher3' package and the SQLCipher C library.\n"
            "  macOS:  brew install sqlcipher && pip install 'engdbram[encryption]'\n"
            "  Linux:  apt install libsqlcipher-dev && pip install 'engdbram[encryption]'"
        ) from exc


def _apply_key(conn: sqlite3.Connection, key: str) -> None:
    """Set the SQLCipher encryption key as a hex blob (avoids injection risk)."""
    key_hex = key.encode().hex()
    conn.execute(f"PRAGMA key = \"x'{key_hex}'\"")


def _configure_connection(conn: sqlite3.Connection, path: str) -> None:
    """Apply performance PRAGMAs to a freshly opened SQLite connection.

    WAL mode allows readers and writers to run concurrently: observe() and
    recall() no longer block each other, and reflect_async() can write facts
    while the main thread reads.  synchronous=NORMAL is safe under WAL because
    a crash mid-WAL leaves the main database intact; only the uncommitted
    WAL entries are discarded.

    cache_size and temp_store are applied to both file and in-memory databases.
    """
    # WAL is file-only; :memory: uses its own journaling (no-op if set, but skip for clarity).
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL: fsync only at WAL checkpoint, not after every commit.
        # Safe under WAL; would be unsafe under the default DELETE journal.
        conn.execute("PRAGMA synchronous=NORMAL")

    # 32 MB page cache (default is ~2 MB). Keeps hot pages in memory,
    # reducing repeated reads for spreading-activation and decay queries.
    conn.execute("PRAGMA cache_size=-32000")
    # Temporary tables/indices in RAM instead of a temp file on disk.
    conn.execute("PRAGMA temp_store=MEMORY")


class Engram:
    """Single-file cognitive memory store for AI agents.

    Args:
        path: Path to the .engram SQLite file, or ``":memory:"`` for an in-process store.
        embedder_model: fastembed model name used for all embeddings.
        decay_config: Importance decay parameters.
        llm: LLM adapter for reflection (optional).
        agent_id: Scope this instance to a named agent. When set, all writes are
            tagged with this id and reads are filtered to it by default. Pass
            ``cross_agent=True`` to :meth:`recall` to search across all agents.
            Leave as ``None`` for single-agent or unscoped use (backward-compatible).
        events_path: Opt-in destination for an Agent Passport NDJSON event
            log (see :mod:`engram.events`). ``None`` (the default) disables
            event emission entirely -- zero overhead. Falls back to the
            ``ENGRAM_EVENTS_PATH`` environment variable when not given.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        embedder_model: str = DEFAULT_MODEL,
        decay_config: DecayConfig | None = None,
        llm: LLMAdapter | None = None,
        agent_id: str | None = None,
        key: str | None = None,
        events_path: str | Path | None = None,
    ) -> None:
        self._path = str(path)
        self._agent_id = agent_id
        self._key = key

        resolved_events_path = resolve_events_path(events_path)
        self._events: EventLog | None = (
            EventLog(resolved_events_path) if resolved_events_path is not None else None
        )

        _mod: Any = _sqlite_module(key)
        # check_same_thread=False: the connection is shared across threads
        # (reflect_async's background thread, AsyncEngram's pool). Concurrent
        # access is serialised by the Store's re-entrant lock, which makes each
        # DB operation atomic; see engram.store._synchronized.
        raw_conn = _mod.connect(self._path, check_same_thread=False)
        if key is not None:
            _apply_key(cast(sqlite3.Connection, raw_conn), key)
        self._conn: sqlite3.Connection = cast(sqlite3.Connection, raw_conn)
        self._conn.row_factory = _mod.Row
        self._decay_cfg = decay_config or DecayConfig()
        self._llm = llm

        _configure_connection(self._conn, self._path)
        self._embedder = Embedder(embedder_model)
        migrate(self._conn, dim=self._embedder.dim)
        self._store = Store(self._conn, dim=self._embedder.dim, agent_id=agent_id)

        # Serializes reflect(): only one reflection pass runs at a time on
        # this instance, even when reflect_async() is fired twice (or
        # reflect() and reflect_async() race) from different threads. A
        # plain Lock, not RLock: reflect() never calls itself recursively.
        # Per-instance (not a module global) to respect the no-global-state
        # invariant -- each Engram gets its own lock.
        #
        # Lock ordering: this lock is always acquired BEFORE any Store
        # method is called (Store methods take self._store's own lock
        # internally -- see engram.store._synchronized). Nothing in this
        # codebase ever acquires the Store lock first and then tries to
        # acquire this reflect lock, so the ordering is a fixed one-way
        # chain (reflect lock -> store lock), never the reverse. A deadlock
        # would require a cycle in that wait-for relationship; a fixed
        # ordering that's never inverted anywhere cannot produce one.
        self._reflect_lock = threading.Lock()

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
        if self._events is not None:
            self._events.emit(
                "memory_written", self._agent_id, {"memory_id": episode_id, "kind": "episodic"}
            )
        return episode_id

    def observe_many(self, items: list[ObserveInput]) -> list[str]:
        """Record multiple observations in a single transaction.

        Uses batch ONNX inference and a single SQL transaction, giving roughly
        linear throughput independent of batch size rather than paying per-call
        overhead for each observation.

        Args:
            items: Observations to record; each is an :class:`ObserveInput` instance.

        Returns:
            Episode ids in the same order as *items*.
        """
        if not items:
            return []

        from engram.models import Episode

        now = datetime.now(tz=UTC)
        episode_ids = [str(uuid.uuid4()) for _ in items]
        episodes = [
            Episode(
                id=ep_id,
                content=item.content,
                timestamp=now,
                actors=item.actors,
                tags=item.tags,
                salience=item.salience,
                emotional_valence=item.emotional_valence,
            )
            for ep_id, item in zip(episode_ids, items, strict=True)
        ]
        embeddings = self._embedder.embed_batch([item.content for item in items])
        self._store.insert_episode_batch(episodes, embeddings)
        return episode_ids

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def recall(
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
        k_inner: int | None = None,
        candidate_limit: int | None = None,
    ) -> list[SearchResult]:
        """Retrieve the top-k episodes most semantically similar to *query*.

        Args:
            query: Natural-language search query.
            k: Maximum number of results to return.
            mode: Retrieval strategy — ``"cosine"`` (default), ``"spreading"``
                for graph-based recall, or ``"hybrid"`` for BM25 + cosine blend.
            depth: BFS hops; only used when ``mode="spreading"``.
            decay: Activation decay per hop; only used when ``mode="spreading"``.
            vector_weight: Cosine fraction for hybrid mode (default 0.7).
            fts_weight: BM25 fraction for hybrid mode (default 0.3).
            as_of: If set, only episodes with timestamp <= as_of are searched.
            cross_agent: If ``True``, search all agents' episodes regardless of
                the instance's ``agent_id``. Ignored when no ``agent_id`` is set.
            k_inner: Inner vector-index limit for as_of search (defaults to k * 10).
            candidate_limit: Candidate search limit per source for hybrid search (defaults to k * 4).

        Returns:
            List of :class:`SearchResult` ordered by descending score.
        """
        agent_id = None if cross_agent else self._agent_id
        return _recall(
            query,
            k,
            self._store,
            self._embedder,
            mode=mode,
            depth=depth,
            decay=decay,
            vector_weight=vector_weight,
            fts_weight=fts_weight,
            as_of=as_of,
            agent_id=agent_id,
            k_inner=k_inner,
            candidate_limit=candidate_limit,
        )

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
        if self._events is not None:
            self._events.emit(
                "memory_written", self._agent_id, {"memory_id": fact_id, "kind": "semantic"}
            )
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

    def timeline(self, entity: str, *, as_of: datetime | None = None) -> list[Fact]:
        """Return the fact history for *entity* in chronological order.

        Args:
            entity: Subject name (e.g. ``"Ivan"``).
            as_of: If set, return only facts whose validity interval contains
                ``as_of`` (``valid_from <= as_of`` and ``valid_to`` is ``None``
                or ``> as_of``). If ``None`` (default), returns the full
                timeline including superseded facts so callers can see how
                beliefs evolved.

        Returns:
            Facts where ``subject == entity``, sorted by ``valid_from`` ascending.
        """
        if as_of is not None:
            return self._store.get_facts_as_of(entity, as_of)
        return self._store.get_all_facts(entity)

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
                        # Same (subject, predicate, object) is agreement, not a
                        # contradiction (e.g. a fact asserted/extracted twice).
                        if facts[i].object != facts[j].object:
                            pairs.append((facts[i], facts[j]))
        return pairs

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------

    def reflect(self) -> ReflectionRun:
        """Run the reflection loop synchronously.

        Serialized per instance via ``self._reflect_lock``: if another
        thread is already inside this method on the same Engram (e.g. a
        concurrent :meth:`reflect_async`), this call blocks until that pass
        finishes. Without this, two racing calls could both read the same
        "since" watermark, both reprocess the same episode window, and both
        insert -- producing duplicate facts and a self-inflicted
        contradiction between the two runs.

        Returns:
            The completed :class:`ReflectionRun`.
        """
        from engram.reflection import reflect as _reflect

        with self._reflect_lock:
            run = _reflect(
                self._store,
                self._llm,
                self._decay_cfg,
                events=self._events,
                agent_id=self._agent_id,
            )
            if self._events is not None:
                self._events.emit(
                    "reflection_run",
                    self._agent_id,
                    {
                        "reflection_run_id": run.id,
                        "episodes_processed": run.episodes_processed,
                        "facts_extracted": run.facts_extracted,
                        "contradictions_resolved": run.contradictions_resolved,
                    },
                    run_id=run.id,
                )
            return run

    def reflect_async(self) -> ReflectionThread:
        """Run the reflection loop in a background thread.

        Returns:
            A started :class:`ReflectionThread`. Call ``.join()`` to wait for
            completion, then access ``.result`` for the :class:`ReflectionRun`.
        """
        thread = ReflectionThread(self)
        thread.start()
        return thread

    # ------------------------------------------------------------------
    # Forget / erasure
    # ------------------------------------------------------------------

    def forget(self, episode_id: str, *, reason: str | None = None) -> None:
        """Permanently erase a single episode from all memory structures.

        Removes the episode from the vector index, access log, and graph edges.
        This operation is irreversible. Scoped to this instance's agent_id:
        when this instance is agent-scoped, an episode_id belonging to a
        different agent (or with no agent_id at all) is treated exactly like
        an unknown id -- an agent can only forget its OWN episodes, even in
        a DB file shared with other agents. Unscoped instances (no agent_id)
        keep today's behaviour and can forget any episode.

        Args:
            episode_id: Id of the episode to erase (returned by :meth:`observe`).
            reason: Optional note for caller's audit trail (not persisted).

        Raises:
            KeyError: If no episode with this id exists, or it belongs to a
                different agent than this instance's agent_id.
        """
        if not self._store.delete_episode(episode_id, agent_id=self._agent_id):
            raise KeyError(f"Episode not found: {episode_id!r}")
        if self._events is not None:
            self._events.emit(
                "memory_forgotten", self._agent_id, {"memory_id": episode_id, "kind": "episodic"}
            )

    def forget_fact(self, fact_id: str) -> None:
        """Permanently erase a single semantic fact.

        Removes the fact row from the store. This operation is irreversible.

        Args:
            fact_id: Id of the fact to erase (returned by :meth:`assert_fact`
                or produced by :meth:`reflect`).

        Raises:
            KeyError: If no fact with this id exists.
        """
        if not self._store.delete_fact(fact_id):
            raise KeyError(f"Fact not found: {fact_id!r}")
        if self._events is not None:
            self._events.emit(
                "memory_forgotten", self._agent_id, {"memory_id": fact_id, "kind": "semantic"}
            )

    def forget_entity(self, entity_name: str) -> ForgetResult:
        """Erase all stored data about an entity (GDPR right-to-be-forgotten).

        Permanently deletes:

        - Episodes where the entity is listed in ``actors``
        - Facts where the entity appears as subject or object
        - The entity record and all graph edges connected to it

        This operation is irreversible.

        Args:
            entity_name: Canonical name of the entity to erase (e.g. ``"Ivan"``).

        Returns:
            :class:`ForgetResult` with counts of deleted episodes and facts.
        """
        return self._store.forget_entity(entity_name)

    # ------------------------------------------------------------------
    # Multi-agent
    # ------------------------------------------------------------------

    def list_agents(self) -> list[str]:
        """Return all distinct agent_ids that have written to this store.

        Useful for inspecting a shared store and iterating over agents.
        Episodes written without an agent_id are not included.

        Returns:
            Sorted list of agent id strings.
        """
        return self._store.list_agents()

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def export_json(self, dest: str) -> dict[str, Any]:
        """Export the full store to a JSON file.

        Exports episodes, facts, entities, and graph edges. Access log and
        reflection records are omitted (operational data, not portable).

        Args:
            dest: File path to write. Extension ``.json`` recommended.

        Returns:
            The exported document as a plain dict.
        """
        from engram.export import export_json

        return export_json(self, dest)

    def import_json(self, src: str, *, merge: bool = False) -> dict[str, int]:
        """Import from a JSON file produced by :meth:`export_json`.

        Args:
            src: Path to the JSON file.
            merge: If ``True``, skip duplicate ids instead of raising.

        Returns:
            Dict with counts of inserted rows per table:
            ``{"episodes": N, "facts": N, "entities": N, "edges": N}``.
        """
        from engram.export import import_json

        return import_json(self, src, merge=merge)

    def backup(self, dest: str | Path) -> None:
        """Create a consistent hot backup of the store to *dest*.

        Uses SQLite's built-in online backup API, which is safe to call while
        the database is open and actively written to.  The destination file
        is created (or overwritten) as a complete, self-contained copy of the
        current database, including all WAL frames.

        Args:
            dest: File path for the backup. Parent directory must exist.
                  Use a ``.engram`` extension by convention.
        """
        self._store.flush_access_log()
        _mod: Any = _sqlite_module(self._key)
        raw = _mod.connect(str(dest))
        target = cast(sqlite3.Connection, raw)
        if self._key is not None:
            _apply_key(target, self._key)
        try:
            self._conn.backup(target)
        finally:
            target.close()

    def rekey(self, new_key: str | None) -> None:
        """Change or remove the encryption passphrase (SQLCipher only).

        Args:
            new_key: New passphrase, or ``None`` to remove encryption.

        Raises:
            ValueError: If the database was not opened with a key.
        """
        if self._key is None:
            raise ValueError(
                "rekey() is only valid on encrypted databases. "
                "To encrypt a plain database, use export_json() + a new Engram(key=...)."
            )
        if new_key is not None:
            key_hex = new_key.encode().hex()
            self._conn.execute(f"PRAGMA rekey = \"x'{key_hex}'\"")
        else:
            self._conn.execute("PRAGMA rekey = ''")
        self._key = new_key

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def decay(self) -> int:
        """Recompute importance scores for all episodes (synchronous).

        Returns:
            Number of episodes updated.
        """
        return run_decay(self._store, self._decay_cfg)

    def compress(
        self,
        *,
        max_episodes: int = 1000,
        importance_threshold: float = 0.3,
        batch_size: int = 20,
    ) -> CompressionRun:
        """Compress low-importance episodes into LLM-generated summaries.

        Selects episodes whose ``importance_score`` is below
        *importance_threshold*, groups them into batches of *batch_size*,
        calls the LLM to produce a single summary paragraph per batch, stores
        each summary as a new episode (with ``summary_of`` pointing to the
        originals), then hard-deletes the originals.

        No-op if no LLM adapter was provided or the store has fewer than
        *max_episodes* episodes.

        Args:
            max_episodes: Only compress when the store exceeds this count.
                Protects small stores from premature lossy compression.
            importance_threshold: Episodes with ``importance_score`` below
                this value are candidates for compression (default 0.3).
            batch_size: Number of episodes grouped into each summary.

        Returns:
            :class:`CompressionRun` with counts of removed episodes and
            created summaries.
        """
        if self._llm is None:
            return CompressionRun(episodes_removed=0, summaries_created=0)

        total = self._store.episode_count()
        if total <= max_episodes:
            return CompressionRun(episodes_removed=0, summaries_created=0)

        candidates = self._store.get_episodes_below_importance(importance_threshold)
        if not candidates:
            return CompressionRun(episodes_removed=0, summaries_created=0)

        removed = 0
        created = 0
        total_tokens = 0

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            summary_text, tokens = self._llm.summarise(batch)
            total_tokens += tokens

            summary_ep_id = self.observe(
                summary_text,
                tags=["summary"],
                salience=max(ep.salience for ep in batch),
            )
            # Update summary_of field on the new episode
            batch_ids = [ep.id for ep in batch]
            self._store.set_summary_of(summary_ep_id, batch_ids)

            for ep in batch:
                self._store.delete_episode(ep.id)
                removed += 1
            created += 1

        return CompressionRun(
            episodes_removed=removed,
            summaries_created=created,
            model_used=self._llm.model_name,
            cost_tokens=total_tokens,
        )

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush buffered access log entries and close the database connection."""
        self._store.flush_access_log()
        self._conn.close()

    def __enter__(self) -> Engram:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
