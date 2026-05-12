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
    score: float      # cosine similarity in [0, 1], higher is better
    distance: float   # raw L2 distance from vec index
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


@dataclass
class Entity:
    """A named entity (person, place, concept)."""

    id: str
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None


# Convenience type alias used by Store
EmbeddedVector = np.ndarray
