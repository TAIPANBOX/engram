"""Core data types for Engram."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class Episode:
    """A raw observed event stored in the episodic memory."""

    id: str
    content: str
    timestamp: datetime
    actors: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    salience: float = 0.5
    emotional_valence: float = 0.0
    summary_of: list[str] = field(default_factory=list)
    importance_score: float = 1.0

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> Episode:
        """Construct from a DB row (id, content, timestamp, actors, tags, salience, emotional_valence, summary_of, importance_score)."""
        return cls(
            id=row[0],
            content=row[1],
            timestamp=datetime.fromisoformat(row[2]),
            actors=json.loads(row[3] or "[]"),
            tags=json.loads(row[4] or "[]"),
            salience=row[5],
            emotional_valence=row[6],
            summary_of=json.loads(row[7] or "[]"),
            importance_score=float(row[8]) if row[8] is not None else 1.0,
        )


@dataclass
class SearchResult:
    """A retrieval hit returned by recall()."""

    episode: Episode
    score: float  # cosine similarity in [0, 1], higher is better
    distance: float  # raw L2 distance from vec index
    importance: float = 1.0  # cached importance score at time of retrieval


@dataclass
class Fact:
    """A semantic triple with bitemporal validity."""

    id: str
    subject: str
    predicate: str
    object: str
    valid_from: datetime
    valid_to: datetime | None
    recorded_at: datetime
    superseded_at: datetime | None
    superseded_by: str | None
    confidence: float
    derived_from: list[str] = field(default_factory=list)
    extracted_by: str | None = None  # reflection run id that created this fact

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> Fact:
        """Construct from a DB row (id, subject, predicate, object, valid_from, valid_to,
        recorded_at, superseded_at, superseded_by, confidence, derived_from, extracted_by)."""
        return cls(
            id=row[0],
            subject=row[1],
            predicate=row[2],
            object=row[3],
            valid_from=datetime.fromisoformat(row[4]),
            valid_to=datetime.fromisoformat(row[5]) if row[5] else None,
            recorded_at=datetime.fromisoformat(row[6]),
            superseded_at=datetime.fromisoformat(row[7]) if row[7] else None,
            superseded_by=row[8],
            confidence=float(row[9]),
            derived_from=json.loads(row[10] or "[]"),
            extracted_by=row[11],
        )


@dataclass
class ReflectionRun:
    """Audit record for a single reflection pass."""

    id: str
    started_at: datetime
    finished_at: datetime | None = None
    episodes_processed: int = 0
    facts_extracted: int = 0
    contradictions_resolved: int = 0
    model_used: str | None = None
    cost_tokens: int = 0

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> ReflectionRun:
        """Construct from a DB row (id, started_at, finished_at, episodes_processed,
        facts_extracted, contradictions_resolved, model_used, cost_tokens)."""
        return cls(
            id=row[0],
            started_at=datetime.fromisoformat(row[1]),
            finished_at=datetime.fromisoformat(row[2]) if row[2] else None,
            episodes_processed=int(row[3] or 0),
            facts_extracted=int(row[4] or 0),
            contradictions_resolved=int(row[5] or 0),
            model_used=row[6],
            cost_tokens=int(row[7] or 0),
        )


@dataclass
class Entity:
    """A named entity (person, place, concept)."""

    id: str
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> Entity:
        """Construct from a DB row (id, name, type, aliases, first_seen, last_seen)."""
        return cls(
            id=row[0],
            name=row[1],
            type=row[2],
            aliases=json.loads(row[3] or "[]"),
            first_seen=datetime.fromisoformat(row[4]) if row[4] else None,
            last_seen=datetime.fromisoformat(row[5]) if row[5] else None,
        )


@dataclass
class Edge:
    """A directed graph edge between two memory nodes."""

    src_id: str
    dst_id: str
    relation: str
    weight: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime(2000, 1, 1))


@dataclass
class ForgetResult:
    """Result of a forget_entity() call."""

    entity: str
    episodes_deleted: int
    facts_deleted: int


# Convenience type alias used by Store
EmbeddedVector = np.ndarray
