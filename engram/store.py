"""Low-level CRUD operations over the SQLite schema."""

from __future__ import annotations

import contextlib
import functools
import json
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any, TypeVar, cast

import numpy as np

from engram.models import EmbeddedVector, Entity, Episode, Fact, ForgetResult, ReflectionRun

_F = TypeVar("_F", bound=Callable[..., Any])
_C = TypeVar("_C", bound=type)


def _locked(method: _F) -> _F:
    """Wrap a Store method so its whole body runs under ``self._lock``."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return cast(_F, wrapper)


def _synchronized(cls: _C) -> _C:
    """Class decorator: serialise every public method behind a re-entrant lock.

    The single sqlite3 connection is shared across threads (reflect_async runs
    on a background thread; AsyncEngram dispatches to a thread pool). The lock
    makes each Store operation atomic so concurrent callers can't interleave a
    half-written transaction. The slow part of reflection (the LLM call) lives
    in reflection.py, *outside* any Store method, so it never holds the lock.
    """
    for name, attr in list(vars(cls).items()):
        if callable(attr) and not name.startswith("_"):
            setattr(cls, name, _locked(attr))
    return cls


def _serialize(vec: EmbeddedVector) -> bytes:
    return vec.astype(np.float32).tobytes()


def _minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max scale a score map into [0, 1].

    When every score is equal (including the single-candidate case), they are
    all mapped to ``1.0`` rather than ``0.0`` — a lone top hit is the *best*
    match, so collapsing it to zero would let noise outrank it after blending.
    """
    if not scores:
        return {}
    hi = max(scores.values())
    lo = min(scores.values())
    if hi == lo:
        return dict.fromkeys(scores, 1.0)
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


def _iso_utc(dt: datetime) -> str:
    """Serialise a datetime to a canonical UTC isoformat for storage/comparison.

    Timestamps are compared lexicographically as TEXT in SQLite. A naive
    ``datetime`` serialises without the ``+00:00`` offset that aware (UTC)
    timestamps carry, which breaks ``<=`` / ``>`` ordering at the boundary and
    silently corrupts bitemporal ``as_of`` results. Coercing naive values to
    UTC (and converting aware values to UTC) guarantees one comparable format.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _distance_to_score(distance: float) -> float:
    """Convert L2 distance from sqlite-vec to a [0, 1] cosine-like score."""
    return float(1.0 / (1.0 + distance))


def _escape_fts5_query(query: str) -> str:
    """Make an arbitrary user string safe for ``fts5 MATCH``.

    FTS5 treats bare ``*``, ``(``, ``OR``, ``NOT``, ``-``, ``"`` and others as
    operators; raw user input therefore crashes the query parser with
    ``OperationalError``. Each non-empty token is double-quoted as a phrase,
    with embedded quotes doubled. Empty input becomes an empty string that the
    caller can short-circuit on.

    Tokens are joined with ``OR``, not with spaces. A space in FTS5 is an
    implicit AND, which requires every word of the query to appear in one
    episode: for a real question ("what was the name of the restaurant I
    mentioned when we talked about my anniversary dinner?") that is sixteen
    words and never matches, so the BM25 side of hybrid recall returned nothing
    and the blend collapsed to ``vector_weight`` times cosine. Hybrid mode was
    silently identical to cosine for any query longer than a few words, which
    is how it reached a LongMemEval run scoring both modes to three identical
    decimal places.

    OR is also what BM25 is for: it ranks by how many query terms a document
    matches and how rare they are, so a document carrying every term still
    wins and common words contribute almost nothing through their low IDF.
    """
    tokens = [tok.replace('"', '""') for tok in query.split() if tok]
    return " OR ".join(f'"{tok}"' for tok in tokens)


_EPISODE_COLS = (
    "id, content, timestamp, actors, tags, salience, "
    "emotional_valence, summary_of, importance_score, agent_id"
)


_ACCESS_LOG_FLUSH_SIZE = 50  # flush buffer after this many entries

# vec0 refuses a KNN k above this and raises OperationalError. Callers get a
# ValueError naming the limit instead, since k arrives straight from a caller's
# recall(k=...) and an internal sqlite error tells them nothing actionable.
_VEC_MAX_K = 4096


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if k > _VEC_MAX_K:
        raise ValueError(
            f"k={k} exceeds the vector index limit of {_VEC_MAX_K}. "
            f"Recall more narrowly, or page through timeline() / recent_episodes()."
        )


@_synchronized
class Store:
    """Thin wrapper around a sqlite3 connection providing episode persistence.

    All public methods are serialised behind ``self._lock`` (see
    :func:`_synchronized`) so the shared connection is safe to use from the
    reflection thread and the AsyncEngram thread pool concurrently.
    """

    def __init__(self, conn: sqlite3.Connection, dim: int, agent_id: str | None = None) -> None:
        self._conn = conn
        self._dim = dim
        self._agent_id = agent_id
        # Re-entrant: some public methods call sibling public methods.
        self._lock = threading.RLock()
        # Access log entries are buffered and written in a single transaction
        # to avoid one commit per recall result. Flushed on close() or when
        # the buffer reaches _ACCESS_LOG_FLUSH_SIZE.
        self._access_buffer: list[tuple[str, str, str | None, int, str | None]] = []
        # >0 while an outer transaction() block is open; see _commit().
        self._tx_depth = 0

    def _commit(self) -> None:
        """Commit now, unless an outer :meth:`transaction` block is active.

        Write methods call this instead of ``self._conn.commit()`` directly
        so they behave exactly as before when called on their own, but
        become no-op-commit steps when called from inside an outer
        :meth:`transaction` block, which commits (or rolls back) everything
        as one unit when it exits.
        """
        if self._tx_depth == 0:
            self._conn.commit()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """Group a sequence of writes into one all-or-nothing SQL transaction.

        Methods called inside this block (``insert_fact``, ``close_fact``,
        ``find_or_create_entity``, ``insert_edge``, ...) still take
        ``self._lock`` as usual -- it is re-entrant -- but skip their normal
        per-call commit via :meth:`_commit`; nothing is durably written
        until the block exits. On success, a single commit makes every
        write visible at once. On any exception, a single rollback undoes
        every write the block made, including partial Hebbian edge-weight
        upserts from :meth:`insert_edge` (``weight = weight + excluded.weight``)
        that a manual per-row DELETE could not safely reverse: once a
        pre-existing weight and this block's delta are merged into one
        number, they cannot be told apart again.

        The lock is held for the block's entire duration (unlike the
        momentary hold a single method call takes), so a concurrent write
        from another thread cannot interleave into this uncommitted
        transaction and get rolled back (or silently committed) with it.

        Reentrant: a nested ``with store.transaction():`` only commits (or
        rolls back) once, at the outermost exit.
        """
        with self._lock:
            self._tx_depth += 1
            try:
                yield
            except BaseException:
                self._tx_depth -= 1
                if self._tx_depth == 0:
                    self._conn.rollback()
                raise
            else:
                self._tx_depth -= 1
                if self._tx_depth == 0:
                    self._conn.commit()

    def insert_episode(self, ep: Episode, embedding: EmbeddedVector) -> None:
        """Persist an episode and its embedding vector."""
        cur = self._conn.execute(
            """
            INSERT INTO episodes (id, content, timestamp, actors, tags,
                                  salience, emotional_valence, summary_of, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ep.id,
                ep.content,
                _iso_utc(ep.timestamp),
                json.dumps(ep.actors),
                json.dumps(ep.tags),
                ep.salience,
                ep.emotional_valence,
                json.dumps(ep.summary_of),
                self._agent_id,
            ),
        )
        rowid: int = cur.lastrowid  # type: ignore[assignment]
        self._conn.execute(
            "INSERT INTO vec_episodes(rowid, agent_id, ts, embedding) VALUES (?, ?, ?, ?)",
            (rowid, self._agent_id, _iso_utc(ep.timestamp), _serialize(embedding)),
        )
        self._conn.execute(
            "INSERT INTO fts_episodes(rowid, content) VALUES (?, ?)",
            (rowid, ep.content),
        )
        self._conn.commit()

    def get_episode(self, episode_id: str, agent_id: str | None = None) -> Episode | None:
        """Fetch a single episode by id.

        Args:
            agent_id: If set, only return the episode when it is owned by
                this agent; an episode owned by a different agent (or with
                a NULL agent_id) is treated as not found. If None (default),
                matches by id alone regardless of owner.
        """
        if agent_id is not None:
            row: Any = self._conn.execute(
                f"SELECT {_EPISODE_COLS} FROM episodes WHERE id = ? AND agent_id = ?",
                (episode_id, agent_id),
            ).fetchone()
        else:
            row = self._conn.execute(
                f"SELECT {_EPISODE_COLS} FROM episodes WHERE id = ?",
                (episode_id,),
            ).fetchone()
        return Episode.from_row(tuple(row)) if row else None

    def search_episodes(
        self,
        query_embedding: EmbeddedVector,
        k: int,
        agent_id: str | None = None,
    ) -> list[tuple[Episode, float, float]]:
        """Return top-k episodes by vector similarity.

        Args:
            agent_id: If set, restrict results to this agent's episodes.
                      If None, search across all agents.

        Returns list of (episode, score, distance) triples, best first.
        """
        _check_k(k)
        # v.agent_id, not e.agent_id: the constraint has to sit on the vec0
        # table to prune partitions during the scan. Moving it to the joined
        # episodes row would put it back outside the KNN, which is the bug
        # this schema exists to remove.
        agent_clause = "AND v.agent_id = ?" if agent_id is not None else ""
        tail: tuple[Any, ...] = (agent_id,) if agent_id is not None else ()

        rows: list[Any] = self._conn.execute(
            f"""
            SELECT e.id, e.content, e.timestamp, e.actors, e.tags,
                   e.salience, e.emotional_valence, e.summary_of, e.importance_score,
                   e.agent_id, v.distance
            FROM vec_episodes v
            JOIN episodes e ON e.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
              {agent_clause}
            ORDER BY v.distance ASC
            """,
            (_serialize(query_embedding), k, *tail),
        ).fetchall()

        results: list[tuple[Episode, float, float]] = []
        for row in rows:
            ep = Episode.from_row(tuple(row[:10]))
            distance = float(row[10])
            score = _distance_to_score(distance)
            results.append((ep, score, distance))
        return results

    def log_access(
        self, memory_id: str, accessed_at: datetime, query: str | None, rank: int
    ) -> None:
        """Buffer a retrieval event; flushes to DB when buffer is full."""
        self._access_buffer.append(
            (memory_id, accessed_at.isoformat(), query, rank, self._agent_id)
        )
        if len(self._access_buffer) >= _ACCESS_LOG_FLUSH_SIZE:
            self.flush_access_log()

    def flush_access_log(self) -> None:
        """Write all buffered access log entries in a single transaction."""
        if not self._access_buffer:
            return
        self._conn.executemany(
            "INSERT INTO access_log (memory_id, accessed_at, query, rank, agent_id) "
            "VALUES (?, ?, ?, ?, ?)",
            self._access_buffer,
        )
        self._conn.commit()
        self._access_buffer.clear()

    def get_access_stats(self, memory_id: str) -> tuple[int, datetime | None]:
        """Return (access_count, last_accessed_at) including buffered entries."""
        row: Any = self._conn.execute(
            "SELECT COUNT(*), MAX(accessed_at) FROM access_log WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        count = int(row[0]) if row[0] else 0
        last: datetime | None = datetime.fromisoformat(row[1]) if row[1] else None

        # Include entries that are still in the write buffer.
        for entry in self._access_buffer:
            if entry[0] == memory_id:
                count += 1
                entry_dt = datetime.fromisoformat(entry[1])
                if last is None or entry_dt > last:
                    last = entry_dt

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

        Scoped to this store's ``agent_id`` when set, so one agent's reflection
        recomputes only its own importance scores with its own DecayConfig —
        symmetric with the agent-scoped :meth:`prune_episodes`.
        """
        if self._agent_id is not None:
            rows: list[Any] = self._conn.execute(
                "SELECT id, salience, emotional_valence, timestamp FROM episodes "
                "WHERE agent_id = ?",
                (self._agent_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, salience, emotional_valence, timestamp FROM episodes"
            ).fetchall()
        return [
            (row[0], float(row[1]), float(row[2]), datetime.fromisoformat(row[3])) for row in rows
        ]

    def get_all_access_stats(self) -> dict[str, tuple[int, datetime | None]]:
        """Return access stats per episode (incl. buffered entries).

        Scoped to this store's ``agent_id`` when set, to match
        :meth:`get_episodes_for_decay`.
        """
        if self._agent_id is not None:
            rows: list[Any] = self._conn.execute(
                "SELECT memory_id, COUNT(*) AS cnt, MAX(accessed_at) AS last "
                "FROM access_log WHERE agent_id = ? GROUP BY memory_id",
                (self._agent_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT memory_id, COUNT(*) AS cnt, MAX(accessed_at) AS last "
                "FROM access_log GROUP BY memory_id"
            ).fetchall()
        result: dict[str, tuple[int, datetime | None]] = {}
        for row in rows:
            last: datetime | None = datetime.fromisoformat(row[2]) if row[2] else None
            result[str(row[0])] = (int(row[1]), last)

        # Merge buffered entries not yet written to the DB.
        for entry in self._access_buffer:
            mid = entry[0]
            entry_dt = datetime.fromisoformat(entry[1])
            if mid in result:
                cnt, last = result[mid]
                result[mid] = (cnt + 1, entry_dt if last is None or entry_dt > last else last)
            else:
                result[mid] = (1, entry_dt)

        return result

    def batch_update_importance(self, scores: dict[str, float]) -> None:
        """Persist importance scores for multiple episodes in a single transaction."""
        self._conn.executemany(
            "UPDATE episodes SET importance_score = ? WHERE id = ?",
            [(score, ep_id) for ep_id, score in scores.items()],
        )
        self._conn.commit()

    def insert_episode_batch(
        self, episodes: list[Episode], embeddings: list[EmbeddedVector]
    ) -> None:
        """Insert multiple episodes and their vectors in a single transaction."""
        self._conn.executemany(
            "INSERT INTO episodes (id, content, timestamp, actors, tags, "
            "salience, emotional_valence, summary_of, agent_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    ep.id,
                    ep.content,
                    _iso_utc(ep.timestamp),
                    json.dumps(ep.actors),
                    json.dumps(ep.tags),
                    ep.salience,
                    ep.emotional_valence,
                    json.dumps(ep.summary_of),
                    self._agent_id,
                )
                for ep in episodes
            ],
        )
        placeholders = ",".join("?" * len(episodes))
        rowid_rows: list[Any] = self._conn.execute(
            f"SELECT id, rowid FROM episodes WHERE id IN ({placeholders})",
            [ep.id for ep in episodes],
        ).fetchall()
        rowid_map = {str(r[0]): int(r[1]) for r in rowid_rows}
        self._conn.executemany(
            "INSERT INTO vec_episodes(rowid, agent_id, ts, embedding) VALUES (?, ?, ?, ?)",
            [
                (rowid_map[ep.id], self._agent_id, _iso_utc(ep.timestamp), _serialize(emb))
                for ep, emb in zip(episodes, embeddings, strict=True)
            ],
        )
        self._conn.executemany(
            "INSERT INTO fts_episodes(rowid, content) VALUES (?, ?)",
            [(rowid_map[ep.id], ep.content) for ep in episodes],
        )
        self._conn.commit()

    def get_episodes_since(self, since: datetime | None, limit: int = 200) -> list[Episode]:
        """Return episodes created after *since* for this agent, up to *limit*.

        When the store has an agent_id, only that agent's episodes are returned.
        """
        if self._agent_id is not None:
            agent_clause = "AND agent_id = ?"
            agent_params: tuple[Any, ...] = (self._agent_id,)
        else:
            agent_clause = ""
            agent_params = ()

        if since is None:
            rows: list[Any] = self._conn.execute(
                f"SELECT {_EPISODE_COLS} FROM episodes WHERE 1=1 {agent_clause} "
                "ORDER BY timestamp ASC LIMIT ?",
                (*agent_params, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {_EPISODE_COLS} FROM episodes "
                f"WHERE timestamp > ? {agent_clause} "
                "ORDER BY timestamp ASC LIMIT ?",
                (_iso_utc(since), *agent_params, limit),
            ).fetchall()
        return [Episode.from_row(tuple(r)) for r in rows]

    def prune_episodes(self, threshold: float) -> int:
        """Delete episodes whose importance_score is below *threshold*.

        Scoped to this store's agent_id when set, so reflection-driven pruning
        cannot delete another agent's episodes from a shared file. Also cleans
        up the vec0 index, the FTS index, access_log, and any incident edges
        (both src and dst directions).

        Returns the number of pruned episodes.
        """
        if self._agent_id is not None:
            rows: list[Any] = self._conn.execute(
                "SELECT rowid, id FROM episodes WHERE importance_score < ? AND agent_id = ?",
                (threshold, self._agent_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT rowid, id FROM episodes WHERE importance_score < ?", (threshold,)
            ).fetchall()
        if not rows:
            return 0
        rowids = [r[0] for r in rows]
        ids = [r[1] for r in rows]
        rowid_ph = ",".join("?" * len(rowids))
        id_ph = ",".join("?" * len(ids))
        self._conn.execute(f"DELETE FROM vec_episodes WHERE rowid IN ({rowid_ph})", rowids)
        self._conn.execute(f"DELETE FROM fts_episodes WHERE rowid IN ({rowid_ph})", rowids)
        self._conn.execute(f"DELETE FROM access_log WHERE memory_id IN ({id_ph})", ids)
        self._conn.execute(
            f"DELETE FROM edges WHERE src_id IN ({id_ph}) OR dst_id IN ({id_ph})",
            (*ids, *ids),
        )
        self._conn.execute(f"DELETE FROM episodes WHERE id IN ({id_ph})", ids)
        self._conn.commit()
        return len(ids)

    # ------------------------------------------------------------------
    # Forget / erasure
    # ------------------------------------------------------------------

    def _get_episode_rowid(self, episode_id: str, agent_id: str | None = None) -> int | None:
        if agent_id is not None:
            row: Any = self._conn.execute(
                "SELECT rowid FROM episodes WHERE id = ? AND agent_id = ?",
                (episode_id, agent_id),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT rowid FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()
        return int(row[0]) if row else None

    def delete_episode(self, episode_id: str, agent_id: str | None = None) -> bool:
        """Hard-delete an episode and all associated data.

        Removes the row from episodes, vec_episodes (ANN index), access_log,
        and any graph edges sourced from this episode.

        Args:
            agent_id: If set, only delete the episode when it is owned by
                this agent; an episode owned by a different agent (or with
                a NULL agent_id) is left untouched and reported as not
                found. If None (default), matches by id alone regardless
                of owner -- used internally by :meth:`forget_entity`, whose
                cross-agent erasure must reach every agent's episodes in a
                shared store.

        Returns:
            True if the episode existed (and was owned by *agent_id*, when
            given) and was deleted; False if not found.
        """
        rowid = self._get_episode_rowid(episode_id, agent_id)
        if rowid is None:
            return False
        self._conn.execute("DELETE FROM vec_episodes WHERE rowid = ?", (rowid,))
        self._conn.execute("DELETE FROM fts_episodes WHERE rowid = ?", (rowid,))
        self._conn.execute("DELETE FROM access_log WHERE memory_id = ?", (episode_id,))
        self._conn.execute(
            "DELETE FROM edges WHERE src_id = ? OR dst_id = ?", (episode_id, episode_id)
        )
        self._conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
        self._commit()
        return True

    def delete_fact(self, fact_id: str) -> bool:
        """Hard-delete a single fact by id.

        Returns:
            True if the fact existed and was deleted; False if not found.
        """
        cur = self._conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        self._commit()
        return cur.rowcount > 0

    def get_episodes_by_actor(self, actor_name: str) -> list[Episode]:
        """Return all episodes where *actor_name* appears in the actors list.

        Intentionally cross-agent: ``forget_entity`` is a GDPR-style erasure
        that must reach every agent's episodes in a shared store. Selects the
        full 10-column tuple so :meth:`Episode.from_row` populates ``agent_id``.
        """
        rows: list[Any] = self._conn.execute(
            f"SELECT {_EPISODE_COLS} FROM episodes "
            "WHERE EXISTS (SELECT 1 FROM json_each(actors) WHERE value = ?)",
            (actor_name,),
        ).fetchall()
        return [Episode.from_row(tuple(r)) for r in rows]

    def get_facts_by_entity(self, entity_name: str) -> list[Fact]:
        """Return all facts (active or superseded) where entity is subject or object."""
        rows: list[Any] = self._conn.execute(
            f"SELECT {self._FACT_COLS} FROM facts WHERE subject = ? OR object = ?",
            (entity_name, entity_name),
        ).fetchall()
        return [Fact.from_row(tuple(r)) for r in rows]

    def delete_entity(self, entity_name: str) -> bool:
        """Delete an entity record and all graph edges connected to it.

        Returns:
            True if the entity existed; False if not found.
        """
        entity = self.get_entity_by_name(entity_name)
        if entity is None:
            return False
        self._conn.execute(
            "DELETE FROM edges WHERE src_id = ? OR dst_id = ?", (entity.id, entity.id)
        )
        self._conn.execute("DELETE FROM entities WHERE id = ?", (entity.id,))
        self._commit()
        return True

    def forget_entity(self, entity_name: str) -> ForgetResult:
        """Erase all stored data about *entity_name* (GDPR right-to-be-forgotten).

        Deletes:
        - Episodes where the entity is listed in ``actors``
        - Facts where ``subject`` or ``object`` equals the entity name
        - The entity record and all graph edges connected to it

        Erasure is all-or-nothing. Without the transaction the method is a run
        of self-committing deletes, so a crash partway through leaves an
        entity half-forgotten with nothing recording that the erasure never
        finished, which is the one outcome a right-to-be-forgotten claim
        cannot survive.

        Returns:
            :class:`ForgetResult` with counts of deleted episodes and facts.
        """
        with self.transaction():
            episodes = self.get_episodes_by_actor(entity_name)
            episodes_deleted = sum(1 for ep in episodes if self.delete_episode(ep.id))

            facts = self.get_facts_by_entity(entity_name)
            facts_deleted = sum(1 for f in facts if self.delete_fact(f.id))

            self.delete_entity(entity_name)

        return ForgetResult(
            entity=entity_name,
            episodes_deleted=episodes_deleted,
            facts_deleted=facts_deleted,
        )

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    _FACT_COLS = (
        "id, subject, predicate, object, valid_from, valid_to, recorded_at, "
        "superseded_at, superseded_by, confidence, derived_from, extracted_by"
    )

    def insert_fact(self, fact: Fact) -> None:
        """Persist a semantic fact triple."""
        self._conn.execute(
            """
            INSERT INTO facts (id, subject, predicate, object, valid_from, valid_to,
                               recorded_at, superseded_at, superseded_by, confidence,
                               derived_from, extracted_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact.id,
                fact.subject,
                fact.predicate,
                fact.object,
                _iso_utc(fact.valid_from),
                _iso_utc(fact.valid_to) if fact.valid_to else None,
                _iso_utc(fact.recorded_at),
                _iso_utc(fact.superseded_at) if fact.superseded_at else None,
                fact.superseded_by,
                fact.confidence,
                json.dumps(fact.derived_from),
                fact.extracted_by,
            ),
        )
        self._commit()

    def get_fact(self, fact_id: str) -> Fact | None:
        """Fetch a single fact by id."""
        row: Any = self._conn.execute(
            f"SELECT {self._FACT_COLS} FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        return Fact.from_row(tuple(row)) if row else None

    def get_active_facts(self, subject: str, predicate: str) -> list[Fact]:
        """Return currently valid facts for (subject, predicate)."""
        rows: list[Any] = self._conn.execute(
            f"SELECT {self._FACT_COLS} FROM facts "
            "WHERE subject = ? AND predicate = ? "
            "AND valid_to IS NULL AND superseded_at IS NULL",
            (subject, predicate),
        ).fetchall()
        return [Fact.from_row(tuple(r)) for r in rows]

    def get_all_active_facts(self) -> list[Fact]:
        """Return all currently valid facts (for contradictions surface query)."""
        rows: list[Any] = self._conn.execute(
            f"SELECT {self._FACT_COLS} FROM facts WHERE valid_to IS NULL AND superseded_at IS NULL"
        ).fetchall()
        return [Fact.from_row(tuple(r)) for r in rows]

    def get_all_facts(self, subject: str) -> list[Fact]:
        """Return all facts for *subject* regardless of validity."""
        rows: list[Any] = self._conn.execute(
            f"SELECT {self._FACT_COLS} FROM facts WHERE subject = ? ORDER BY valid_from ASC",
            (subject,),
        ).fetchall()
        return [Fact.from_row(tuple(r)) for r in rows]

    def close_fact(self, fact_id: str, valid_to: datetime, superseded_by: str) -> None:
        """Mark a fact as superseded by setting its validity end."""
        self._conn.execute(
            "UPDATE facts SET valid_to = ?, superseded_by = ?, superseded_at = ? WHERE id = ?",
            (_iso_utc(valid_to), superseded_by, _iso_utc(valid_to), fact_id),
        )
        self._commit()

    # ------------------------------------------------------------------
    # Reflections
    # ------------------------------------------------------------------

    _REFLECTION_COLS = (
        "id, started_at, finished_at, episodes_processed, facts_extracted, "
        "contradictions_resolved, model_used, cost_tokens"
    )

    def insert_reflection(self, run: ReflectionRun) -> None:
        """Persist a new reflection run record."""
        self._conn.execute(
            """
            INSERT INTO reflections (id, started_at, finished_at, episodes_processed,
                                     facts_extracted, contradictions_resolved,
                                     model_used, cost_tokens, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                run.started_at.isoformat(),
                run.finished_at.isoformat() if run.finished_at else None,
                run.episodes_processed,
                run.facts_extracted,
                run.contradictions_resolved,
                run.model_used,
                run.cost_tokens,
                self._agent_id,
            ),
        )
        self._conn.commit()

    def update_reflection(
        self,
        run_id: str,
        finished_at: datetime,
        episodes_processed: int,
        facts_extracted: int,
        contradictions_resolved: int,
        cost_tokens: int,
    ) -> None:
        """Update a reflection run once it completes."""
        self._conn.execute(
            """
            UPDATE reflections
            SET finished_at = ?, episodes_processed = ?, facts_extracted = ?,
                contradictions_resolved = ?, cost_tokens = ?
            WHERE id = ?
            """,
            (
                finished_at.isoformat(),
                episodes_processed,
                facts_extracted,
                contradictions_resolved,
                cost_tokens,
                run_id,
            ),
        )
        self._conn.commit()

    def get_last_reflection(self) -> ReflectionRun | None:
        """Return the most recently started reflection run for this agent, or None."""
        if self._agent_id is not None:
            row: Any = self._conn.execute(
                f"SELECT {self._REFLECTION_COLS} FROM reflections "
                "WHERE agent_id = ? ORDER BY started_at DESC LIMIT 1",
                (self._agent_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                f"SELECT {self._REFLECTION_COLS} FROM reflections ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        return ReflectionRun.from_row(tuple(row)) if row else None

    def get_last_finished_reflection(self) -> ReflectionRun | None:
        """Return the most recently *completed* reflection run for this agent.

        Unlike :meth:`get_last_reflection`, this ignores runs whose
        ``finished_at`` is NULL. The incremental-reflection window keys off this
        so a run that aborted (e.g. an LLM error) does not advance ``since`` and
        silently skip the episodes it never processed.
        """
        if self._agent_id is not None:
            row: Any = self._conn.execute(
                f"SELECT {self._REFLECTION_COLS} FROM reflections "
                "WHERE agent_id = ? AND finished_at IS NOT NULL "
                "ORDER BY started_at DESC LIMIT 1",
                (self._agent_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                f"SELECT {self._REFLECTION_COLS} FROM reflections "
                "WHERE finished_at IS NOT NULL ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        return ReflectionRun.from_row(tuple(row)) if row else None

    def delete_reflection(self, run_id: str) -> None:
        """Remove a reflection run record (used to roll back an aborted run)."""
        self._conn.execute("DELETE FROM reflections WHERE id = ?", (run_id,))
        self._conn.commit()

    def get_reflection_by_id(self, run_id: str) -> ReflectionRun | None:
        """Fetch a specific reflection run by id."""
        row: Any = self._conn.execute(
            f"SELECT {self._REFLECTION_COLS} FROM reflections WHERE id = ?", (run_id,)
        ).fetchone()
        return ReflectionRun.from_row(tuple(row)) if row else None

    def episode_count(self) -> int:
        if self._agent_id is not None:
            row: Any = self._conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE agent_id = ?", (self._agent_id,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return int(row[0])

    def get_episodes_below_importance(self, threshold: float) -> list[Episode]:
        """Return episodes whose importance_score < threshold, oldest first."""
        if self._agent_id is not None:
            rows: list[Any] = self._conn.execute(
                f"SELECT {_EPISODE_COLS} FROM episodes "
                "WHERE importance_score < ? AND agent_id = ? "
                "ORDER BY timestamp ASC",
                (threshold, self._agent_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {_EPISODE_COLS} FROM episodes "
                "WHERE importance_score < ? ORDER BY timestamp ASC",
                (threshold,),
            ).fetchall()
        return [Episode.from_row(tuple(r)) for r in rows]

    def set_summary_of(self, episode_id: str, source_ids: list[str]) -> None:
        """Update the summary_of field of an existing episode."""
        self._conn.execute(
            "UPDATE episodes SET summary_of = ? WHERE id = ?",
            (json.dumps(source_ids), episode_id),
        )
        self._conn.commit()

    def vec_count(self) -> int:
        row: Any = self._conn.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()
        return int(row[0])

    def fact_count(self) -> int:
        row: Any = self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()
        return int(row[0])

    def active_fact_count(self) -> int:
        row: Any = self._conn.execute(
            "SELECT COUNT(*) FROM facts WHERE valid_to IS NULL AND superseded_at IS NULL"
        ).fetchone()
        return int(row[0])

    def entity_count(self) -> int:
        row: Any = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()
        return int(row[0])

    def reflection_count(self) -> int:
        row: Any = self._conn.execute("SELECT COUNT(*) FROM reflections").fetchone()
        return int(row[0])

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    _ENTITY_COLS = "id, name, type, aliases, first_seen, last_seen"

    def find_or_create_entity(self, name: str, entity_type: str, now: datetime) -> Entity:
        """Return existing entity by name, or create one.

        Always updates last_seen to *now*.
        """
        existing = self.get_entity_by_name(name)
        if existing is not None:
            self._conn.execute(
                "UPDATE entities SET last_seen = ? WHERE id = ?",
                (now.isoformat(), existing.id),
            )
            self._commit()
            existing.last_seen = now
            return existing
        entity_id = str(uuid.uuid4())
        self._conn.execute(
            f"INSERT INTO entities ({self._ENTITY_COLS}) VALUES (?, ?, ?, ?, ?, ?)",
            (entity_id, name, entity_type, "[]", now.isoformat(), now.isoformat()),
        )
        self._commit()
        return Entity(
            id=entity_id,
            name=name,
            type=entity_type,
            aliases=[],
            first_seen=now,
            last_seen=now,
        )

    def get_entity_by_name(self, name: str) -> Entity | None:
        """Fetch a single entity by its canonical name."""
        row: Any = self._conn.execute(
            f"SELECT {self._ENTITY_COLS} FROM entities WHERE name = ?", (name,)
        ).fetchone()
        return Entity.from_row(tuple(row)) if row else None

    def get_all_entities(self) -> list[Entity]:
        """Return all known entities."""
        rows: list[Any] = self._conn.execute(f"SELECT {self._ENTITY_COLS} FROM entities").fetchall()
        return [Entity.from_row(tuple(r)) for r in rows]

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def insert_edge(
        self,
        src_id: str,
        dst_id: str,
        relation: str,
        weight: float,
        created_at: datetime,
    ) -> None:
        """Insert or accumulate a graph edge (Hebbian weight reinforcement).

        The edge is stamped with this store's ``agent_id`` so spreading
        activation can stay within an agent's own episode graph.
        """
        self._conn.execute(
            """
            INSERT INTO edges (src_id, dst_id, relation, weight, created_at, agent_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(src_id, dst_id, relation)
            DO UPDATE SET weight = weight + excluded.weight
            """,
            (src_id, dst_id, relation, weight, created_at.isoformat(), self._agent_id),
        )
        self._commit()

    def get_neighbors(self, node_id: str) -> list[tuple[str, float]]:
        """Return (neighbor_id, weight) from both directions of all edges.

        When this store is scoped to an ``agent_id``, only that agent's edges
        are traversed, so spreading activation can't hop through another
        agent's episodes. An unscoped store sees every edge.
        """
        if self._agent_id is not None:
            rows: list[Any] = self._conn.execute(
                """
                SELECT dst_id, weight FROM edges WHERE src_id = ? AND agent_id = ?
                UNION
                SELECT src_id, weight FROM edges WHERE dst_id = ? AND agent_id = ?
                """,
                (node_id, self._agent_id, node_id, self._agent_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT dst_id, weight FROM edges WHERE src_id = ?
                UNION
                SELECT src_id, weight FROM edges WHERE dst_id = ?
                """,
                (node_id, node_id),
            ).fetchall()
        return [(str(r[0]), float(r[1])) for r in rows]

    def get_episodes_by_ids(self, ids: list[str], agent_id: str | None = None) -> list[Episode]:
        """Bulk-fetch episodes by id list; silently skips unknown ids.

        Args:
            agent_id: If set, restrict to this agent's episodes.
        """
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        if agent_id is not None:
            rows: list[Any] = self._conn.execute(
                f"SELECT {_EPISODE_COLS} FROM episodes "
                f"WHERE id IN ({placeholders}) AND agent_id = ?",
                (*ids, agent_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {_EPISODE_COLS} FROM episodes WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        return [Episode.from_row(tuple(r)) for r in rows]

    # ------------------------------------------------------------------
    # Bitemporal queries
    # ------------------------------------------------------------------

    def search_episodes_as_of(
        self,
        query_embedding: EmbeddedVector,
        k: int,
        as_of: datetime,
        agent_id: str | None = None,
    ) -> list[tuple[Episode, float, float]]:
        """Like search_episodes but restricted to episodes with timestamp <= as_of.

        Both filters are vec0 constraints (``v.ts`` is a metadata column,
        ``v.agent_id`` a partition key), so they are evaluated inside the KNN
        scan and k counts only rows that already passed them.

        Args:
            agent_id: If set, restrict to this agent's episodes.
        """
        _check_k(k)
        agent_clause = "AND v.agent_id = ?" if agent_id is not None else ""
        tail: tuple[Any, ...] = (
            (_iso_utc(as_of), agent_id) if agent_id is not None else (_iso_utc(as_of),)
        )

        sql = f"""
            SELECT e.id, e.content, e.timestamp, e.actors, e.tags,
                   e.salience, e.emotional_valence, e.summary_of, e.importance_score,
                   e.agent_id, v.distance
            FROM vec_episodes v
            JOIN episodes e ON e.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
              AND v.ts <= ?
              {agent_clause}
            ORDER BY v.distance ASC
        """

        rows: list[Any] = self._conn.execute(
            sql, (_serialize(query_embedding), k, *tail)
        ).fetchall()
        results: list[tuple[Episode, float, float]] = []
        for row in rows:
            ep = Episode.from_row(tuple(row[:10]))
            distance = float(row[10])
            score = _distance_to_score(distance)
            results.append((ep, score, distance))
        return results

    def search_episodes_hybrid(
        self,
        query: str,
        query_embedding: EmbeddedVector,
        k: int,
        *,
        vector_weight: float = 0.5,
        fts_weight: float = 0.5,
        agent_id: str | None = None,
        candidate_limit: int | None = None,
        as_of: datetime | None = None,
    ) -> list[tuple[Episode, float, float]]:
        """Return top-k episodes using BM25 + cosine blended score.

        Pulls up to ``candidate_limit`` candidates from each source, normalises both
        score distributions to [0, 1], then returns the top-k by the weighted
        combination ``vector_weight * cosine + fts_weight * bm25``.

        Args:
            query: Raw text used for the FTS5 BM25 search.
            query_embedding: Pre-computed embedding for vector search.
            k: Maximum number of results to return.
            vector_weight: Weight applied to the normalised cosine score.
            fts_weight: Weight applied to the normalised BM25 score.
            agent_id: If set, restrict to this agent's episodes.
            candidate_limit: Candidate search limit per source (defaults to k * 4).
            as_of: If set, only episodes with ``timestamp <= as_of`` are
                considered in both the vector and FTS candidate pools.
        """
        _check_k(k)
        if candidate_limit is None:
            candidate_limit = k * 4
        # Clamped, not validated. The pool size is derived (k * 4 by default),
        # so checking it against the vec0 ceiling made hybrid refuse a k of
        # 1025 while cosine accepted 4096, and reported the failure as
        # "k=4100", a number the caller never passed. A smaller pool degrades
        # the blend a little; refusing the query outright is worse.
        candidate_limit = min(candidate_limit, _VEC_MAX_K)

        # The two sources filter through different columns: vec0 needs its own
        # partition key and metadata column so the constraints run inside the
        # KNN scan, while FTS joins episodes and filters there, ahead of LIMIT.
        vec_clause = ""
        vec_params: tuple[Any, ...] = ()
        fts_clause = ""
        fts_params: tuple[Any, ...] = ()
        if agent_id is not None:
            vec_clause += " AND v.agent_id = ?"
            vec_params = (*vec_params, agent_id)
            fts_clause += " AND e.agent_id = ?"
            fts_params = (*fts_params, agent_id)
        if as_of is not None:
            vec_clause += " AND v.ts <= ?"
            vec_params = (*vec_params, _iso_utc(as_of))
            fts_clause += " AND e.timestamp <= ?"
            fts_params = (*fts_params, _iso_utc(as_of))

        # --- Vector candidates ---
        vec_rows: list[Any] = self._conn.execute(
            f"""
            SELECT e.id, e.content, e.timestamp, e.actors, e.tags,
                   e.salience, e.emotional_valence, e.summary_of, e.importance_score,
                   e.agent_id, v.distance
            FROM vec_episodes v
            JOIN episodes e ON e.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
              {vec_clause}
            ORDER BY v.distance ASC
            """,
            (_serialize(query_embedding), candidate_limit, *vec_params),
        ).fetchall()

        # distance → cosine score, then normalise to [0, 1] across candidates
        vec_scores: dict[str, float] = {
            str(row[0]): _distance_to_score(float(row[10])) for row in vec_rows
        }
        vec_episodes: dict[str, Episode] = {
            str(row[0]): Episode.from_row(tuple(row[:10])) for row in vec_rows
        }

        norm_vec = _minmax_normalize(vec_scores)

        # --- FTS candidates (BM25, lower is better — negate to make higher = better) ---
        fts_query = _escape_fts5_query(query)
        fts_rows: list[Any] = self._conn.execute(
            f"""
            SELECT e.id, -bm25(fts_episodes) AS bm25_score
            FROM fts_episodes
            JOIN episodes e ON e.rowid = fts_episodes.rowid
            WHERE fts_episodes MATCH ?
              {fts_clause}
            ORDER BY bm25_score DESC
            LIMIT ?
            """,
            (fts_query, *fts_params, candidate_limit),
        ).fetchall()

        fts_scores: dict[str, float] = {str(row[0]): float(row[1]) for row in fts_rows}
        norm_fts = _minmax_normalize(fts_scores)

        # --- Merge and rank ---
        all_ids = set(norm_vec) | set(norm_fts)
        blended: list[tuple[str, float]] = [
            (
                eid,
                vector_weight * norm_vec.get(eid, 0.0) + fts_weight * norm_fts.get(eid, 0.0),
            )
            for eid in all_ids
        ]
        blended.sort(key=lambda x: x[1], reverse=True)

        results: list[tuple[Episode, float, float]] = []
        for eid, blend_score in blended[:k]:
            if eid in vec_episodes:
                ep = vec_episodes[eid]
            else:
                ep_row: Any = self._conn.execute(
                    f"SELECT {_EPISODE_COLS} FROM episodes WHERE id = ?", (eid,)
                ).fetchone()
                if ep_row is None:
                    continue
                ep = Episode.from_row(tuple(ep_row))
            raw_dist = 1.0 / blend_score - 1.0 if blend_score > 0 else float("inf")
            results.append((ep, blend_score, raw_dist))
        return results

    def get_facts_as_of(self, subject: str, as_of: datetime) -> list[Fact]:
        """Return facts for *subject* that were valid at *as_of*.

        A fact is valid at T when valid_from <= T and (valid_to IS NULL or valid_to > T).
        """
        rows: list[Any] = self._conn.execute(
            f"SELECT {self._FACT_COLS} FROM facts "
            "WHERE subject = ? "
            "  AND valid_from <= ? "
            "  AND (valid_to IS NULL OR valid_to > ?)"
            " ORDER BY valid_from ASC",
            (subject, _iso_utc(as_of), _iso_utc(as_of)),
        ).fetchall()
        return [Fact.from_row(tuple(r)) for r in rows]

    # ------------------------------------------------------------------
    # Multi-agent
    # ------------------------------------------------------------------

    def list_agents(self) -> list[str]:
        """Return all distinct agent_ids that have written episodes to this store.

        Episodes with no agent (agent_id IS NULL) are excluded.
        """
        rows: list[Any] = self._conn.execute(
            "SELECT DISTINCT agent_id FROM episodes WHERE agent_id IS NOT NULL ORDER BY agent_id"
        ).fetchall()
        return [str(r[0]) for r in rows]
