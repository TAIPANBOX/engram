"""Low-level CRUD operations over the SQLite schema."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any

import numpy as np

from engram.models import EmbeddedVector, Entity, Episode, Fact, ForgetResult, ReflectionRun


def _serialize(vec: EmbeddedVector) -> bytes:
    return vec.astype(np.float32).tobytes()


def _distance_to_score(distance: float) -> float:
    """Convert L2 distance from sqlite-vec to a [0, 1] cosine-like score."""
    return float(1.0 / (1.0 + distance))


_EPISODE_COLS = (
    "id, content, timestamp, actors, tags, salience, "
    "emotional_valence, summary_of, importance_score, agent_id"
)


class Store:
    """Thin wrapper around a sqlite3 connection providing episode persistence."""

    def __init__(self, conn: sqlite3.Connection, dim: int, agent_id: str | None = None) -> None:
        self._conn = conn
        self._dim = dim
        self._agent_id = agent_id

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
                ep.timestamp.isoformat(),
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
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
            (rowid, _serialize(embedding)),
        )
        self._conn.commit()

    def get_episode(self, episode_id: str) -> Episode | None:
        """Fetch a single episode by id."""
        row: Any = self._conn.execute(
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
        if agent_id is not None:
            agent_clause = "AND e.agent_id = ?"
            params: tuple[Any, ...] = (_serialize(query_embedding), k, agent_id)
        else:
            agent_clause = ""
            params = (_serialize(query_embedding), k)

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
            params,
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
        """Record a retrieval event in the access log."""
        self._conn.execute(
            "INSERT INTO access_log (memory_id, accessed_at, query, rank, agent_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (memory_id, accessed_at.isoformat(), query, rank, self._agent_id),
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
        """Return minimal episode data needed by the decay job (all agents)."""
        rows: list[Any] = self._conn.execute(
            "SELECT id, salience, emotional_valence, timestamp FROM episodes"
        ).fetchall()
        return [
            (row[0], float(row[1]), float(row[2]), datetime.fromisoformat(row[3])) for row in rows
        ]

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
                (since.isoformat(), *agent_params, limit),
            ).fetchall()
        return [Episode.from_row(tuple(r)) for r in rows]

    def prune_episodes(self, threshold: float) -> int:
        """Delete episodes whose importance_score is below *threshold*.

        Returns the number of pruned episodes.
        """
        rows: list[Any] = self._conn.execute(
            "SELECT rowid, id FROM episodes WHERE importance_score < ?", (threshold,)
        ).fetchall()
        if not rows:
            return 0
        rowids = [r[0] for r in rows]
        ids = [r[1] for r in rows]
        rowid_ph = ",".join("?" * len(rowids))
        id_ph = ",".join("?" * len(ids))
        self._conn.execute(f"DELETE FROM vec_episodes WHERE rowid IN ({rowid_ph})", rowids)
        self._conn.execute(f"DELETE FROM access_log WHERE memory_id IN ({id_ph})", ids)
        self._conn.execute(f"DELETE FROM edges WHERE src_id IN ({id_ph})", ids)
        self._conn.execute(f"DELETE FROM episodes WHERE id IN ({id_ph})", ids)
        self._conn.commit()
        return len(ids)

    # ------------------------------------------------------------------
    # Forget / erasure
    # ------------------------------------------------------------------

    def _get_episode_rowid(self, episode_id: str) -> int | None:
        row: Any = self._conn.execute(
            "SELECT rowid FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        return int(row[0]) if row else None

    def delete_episode(self, episode_id: str) -> bool:
        """Hard-delete an episode and all associated data.

        Removes the row from episodes, vec_episodes (ANN index), access_log,
        and any graph edges sourced from this episode.

        Returns:
            True if the episode existed and was deleted; False if not found.
        """
        rowid = self._get_episode_rowid(episode_id)
        if rowid is None:
            return False
        self._conn.execute("DELETE FROM vec_episodes WHERE rowid = ?", (rowid,))
        self._conn.execute("DELETE FROM access_log WHERE memory_id = ?", (episode_id,))
        self._conn.execute(
            "DELETE FROM edges WHERE src_id = ? OR dst_id = ?", (episode_id, episode_id)
        )
        self._conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
        self._conn.commit()
        return True

    def delete_fact(self, fact_id: str) -> bool:
        """Hard-delete a single fact by id.

        Returns:
            True if the fact existed and was deleted; False if not found.
        """
        cur = self._conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def get_episodes_by_actor(self, actor_name: str) -> list[Episode]:
        """Return all episodes where *actor_name* appears in the actors list."""
        rows: list[Any] = self._conn.execute(
            "SELECT id, content, timestamp, actors, tags, salience, emotional_valence, "
            "summary_of, importance_score FROM episodes "
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
        self._conn.commit()
        return True

    def forget_entity(self, entity_name: str) -> ForgetResult:
        """Erase all stored data about *entity_name* (GDPR right-to-be-forgotten).

        Deletes:
        - Episodes where the entity is listed in ``actors``
        - Facts where ``subject`` or ``object`` equals the entity name
        - The entity record and all graph edges connected to it

        Returns:
            :class:`ForgetResult` with counts of deleted episodes and facts.
        """
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
                fact.valid_from.isoformat(),
                fact.valid_to.isoformat() if fact.valid_to else None,
                fact.recorded_at.isoformat(),
                fact.superseded_at.isoformat() if fact.superseded_at else None,
                fact.superseded_by,
                fact.confidence,
                json.dumps(fact.derived_from),
                fact.extracted_by,
            ),
        )
        self._conn.commit()

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
            (valid_to.isoformat(), superseded_by, valid_to.isoformat(), fact_id),
        )
        self._conn.commit()

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
            self._conn.commit()
            existing.last_seen = now
            return existing
        entity_id = str(uuid.uuid4())
        self._conn.execute(
            f"INSERT INTO entities ({self._ENTITY_COLS}) VALUES (?, ?, ?, ?, ?, ?)",
            (entity_id, name, entity_type, "[]", now.isoformat(), now.isoformat()),
        )
        self._conn.commit()
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
        """Insert or accumulate a graph edge (Hebbian weight reinforcement)."""
        self._conn.execute(
            """
            INSERT INTO edges (src_id, dst_id, relation, weight, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(src_id, dst_id, relation)
            DO UPDATE SET weight = weight + excluded.weight
            """,
            (src_id, dst_id, relation, weight, created_at.isoformat()),
        )
        self._conn.commit()

    def get_neighbors(self, node_id: str) -> list[tuple[str, float]]:
        """Return (neighbor_id, weight) from both directions of all edges."""
        rows: list[Any] = self._conn.execute(
            """
            SELECT dst_id, weight FROM edges WHERE src_id = ?
            UNION
            SELECT src_id, weight FROM edges WHERE dst_id = ?
            """,
            (node_id, node_id),
        ).fetchall()
        return [(str(r[0]), float(r[1])) for r in rows]

    def get_episodes_by_ids(
        self, ids: list[str], agent_id: str | None = None
    ) -> list[Episode]:
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

        Uses k*10 as the inner vector-index limit so there are enough candidates
        after the timestamp filter; returns at most k results.

        Args:
            agent_id: If set, restrict to this agent's episodes.
        """
        k_inner = k * 10
        if agent_id is not None:
            agent_clause = "AND e.agent_id = ?"
            params: tuple[Any, ...] = (_serialize(query_embedding), k_inner, as_of.isoformat(), agent_id)
        else:
            agent_clause = ""
            params = (_serialize(query_embedding), k_inner, as_of.isoformat())

        rows: list[Any] = self._conn.execute(
            f"""
            SELECT e.id, e.content, e.timestamp, e.actors, e.tags,
                   e.salience, e.emotional_valence, e.summary_of, e.importance_score,
                   e.agent_id, v.distance
            FROM vec_episodes v
            JOIN episodes e ON e.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
              AND e.timestamp <= ?
              {agent_clause}
            ORDER BY v.distance ASC
            """,
            params,
        ).fetchall()
        results: list[tuple[Episode, float, float]] = []
        for row in rows[:k]:
            ep = Episode.from_row(tuple(row[:10]))
            distance = float(row[10])
            score = _distance_to_score(distance)
            results.append((ep, score, distance))
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
            (subject, as_of.isoformat(), as_of.isoformat()),
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
            "SELECT DISTINCT agent_id FROM episodes "
            "WHERE agent_id IS NOT NULL ORDER BY agent_id"
        ).fetchall()
        return [str(r[0]) for r in rows]
