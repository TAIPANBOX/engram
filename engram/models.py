"""Core data types for Engram."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

# Where a memory came from. A writer declares one of the three; a row written
# before the field existed has no value, and reads as UNKNOWN_PROVENANCE.
PROVENANCES: tuple[str, ...] = ("user", "tool", "external")
UNKNOWN_PROVENANCE = "unknown"
# What a write records when its caller names no source. A caller who says
# nothing is not assumed to be the user, so the default is the least trusted
# of the three declared values.
DEFAULT_PROVENANCE = "external"
# Least trusted first. Used to read a fact off the episodes it came from.
_TRUST_ORDER: tuple[str, ...] = (UNKNOWN_PROVENANCE, "external", "tool", "user")
# The one source a reader may take as the person's own words.
TRUSTED_PROVENANCE = "user"


def check_provenance(value: str) -> str:
    """Return *value* if it is one of the three declared sources, else raise."""
    if value not in PROVENANCES:
        raise ValueError(f"provenance must be one of {PROVENANCES}, got {value!r}")
    return value


def check_fact_provenance(value: str) -> str:
    """Return *value* if a fact may carry it, else raise.

    A fact may also carry ``unknown``: a reflection over episodes whose source
    was never recorded reads as the least trusted of them, and a fact asserted
    with no declared source is stored as unknown. A caller still cannot
    *declare* ``unknown`` (see :func:`check_provenance`); this is for rows
    read back from a document.
    """
    if value not in (*PROVENANCES, UNKNOWN_PROVENANCE):
        raise ValueError(
            f"a fact's provenance must be one of {(*PROVENANCES, UNKNOWN_PROVENANCE)}, "
            f"got {value!r}"
        )
    return value


def least_trusted(values: Iterable[str]) -> str:
    """The least trusted of *values*. No values at all reads as unknown."""
    ranked = list(values)
    if not ranked:
        return UNKNOWN_PROVENANCE

    def rank(value: str) -> int:
        # A value outside the known set (a hand-edited row) is treated as the
        # least trusted, never as trusted.
        return _TRUST_ORDER.index(value) if value in _TRUST_ORDER else 0

    return min(ranked, key=rank)


def is_untrusted(provenance: str, quarantined: bool) -> bool:
    """Whether reading a memory means reading content the user did not write.

    Anything but a ``user`` memory that was not quarantined counts: a tool's
    output, an external page, another agent's write, a row whose source was
    never recorded, and a write made after its session read any of those.
    """
    return provenance != TRUSTED_PROVENANCE or quarantined


def is_human(provenance: str, quarantined: bool) -> bool:
    """Whether a memory is the person's own word, as far as engram can tell.

    The same test as :func:`is_untrusted`, read the other way: provenance
    ``user`` and not written after its session read untrusted content.
    """
    return not is_untrusted(provenance, quarantined)


def may_displace(incoming: tuple[str, bool], held: tuple[str, bool]) -> bool:
    """Whether a write may close or override a fact that is already held.

    Each side is ``(provenance, quarantined)``. The one thing this refuses is
    a write that is not the person's own acting on a fact that is. Between
    anything else, recency decides, as it always did: two facts that are both
    the person's own, or two that are neither, are not ranked by this.
    """
    return not (is_human(*held) and not is_human(*incoming))


def _legacy_flag(row: tuple[Any, ...], index: int) -> int | None:
    """The flag at *index*, or ``None`` when the row predates the column.

    A missing column and a NULL read the same way on purpose: both describe a
    row written before engram recorded the flag, and both must read as the
    least trusted answer rather than as a clean one.
    """
    if len(row) <= index or row[index] is None:
        return None
    return int(row[index])


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
    agent_id: str | None = None
    provenance: str = UNKNOWN_PROVENANCE
    # Written after its session read untrusted content (see engram.mcp_server).
    quarantined: bool = False

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> Episode:
        """Construct from a DB row (id, content, timestamp, actors, tags, salience,
        emotional_valence, summary_of, importance_score[, agent_id[, provenance
        [, quarantined]]]).

        A NULL provenance is a row written before the column existed, and reads
        as unknown, never as a trusted source.

        A NULL ``quarantined`` is a row written before engram set provenance
        itself, when any caller could declare any source, ``user`` included.
        Such a row reads as quarantined and its provenance as unknown, whatever
        it recorded: the stored value is kept on disk, untouched, and is the
        caller's claim rather than anything engram established.
        """
        quarantined = _legacy_flag(row, 11)
        legacy = quarantined is None
        recorded = row[10] if len(row) > 10 else None
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
            agent_id=row[9] if len(row) > 9 else None,
            provenance=(UNKNOWN_PROVENANCE if legacy or recorded is None else recorded),
            quarantined=legacy or bool(quarantined),
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
    # The agent that wrote the fact, None for an unscoped writer or a fact
    # written before ownership was recorded. Only the owner may delete it.
    agent_id: str | None = None
    # Visible to every agent in the file, or only to its owner.
    shared: bool = True
    # Written after its session read untrusted content.
    quarantined: bool = False
    # Where the fact came from, as recorded when it was written: the least
    # trusted source of the episodes a reflection extracted it from, or what the
    # caller declared to ``assert_fact`` (``unknown`` when it declared none).
    # ``None`` is a row written before engram recorded this; it is then read
    # off the episodes in ``derived_from``, which is how it always was.
    provenance: str | None = None
    # Provenance of whoever refuted the fact, ``None`` while it stands. A
    # refuted fact is closed (``valid_to`` and ``superseded_at`` are set) and
    # never deleted.
    refuted_by: str | None = None
    # Ids of the refuted facts this one rests on, directly or through other
    # facts. A marked fact is not closed: it may hold on other grounds.
    undermined_by: list[str] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> Fact:
        """Construct from a DB row (id, subject, predicate, object, valid_from, valid_to,
        recorded_at, superseded_at, superseded_by, confidence, derived_from, extracted_by
        [, agent_id, shared, quarantined[, provenance, refuted_by, undermined_by]]).

        A fact written before ownership was recorded has NULL in all three. It
        stays visible to every agent, as every fact was then, and reads as
        quarantined, since nothing recorded what its writer had read.

        A fact written before provenance, refutation and marks were recorded
        has NULL in those three, and reads as no recorded provenance, not
        refuted and not marked.
        """
        shared = _legacy_flag(row, 13)
        quarantined = _legacy_flag(row, 14)
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
            agent_id=row[12] if len(row) > 12 else None,
            shared=shared is None or bool(shared),
            quarantined=quarantined is None or bool(quarantined),
            provenance=row[15] if len(row) > 15 else None,
            refuted_by=row[16] if len(row) > 16 else None,
            undermined_by=json.loads(row[17] or "[]") if len(row) > 17 else [],
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


@dataclass
class RefuteResult:
    """Result of a refute_fact() call."""

    fact_id: str
    # Facts newly marked as resting on the refuted one, at any depth. Only the
    # ones the caller may see: a fact another agent keeps private is marked too
    # and is not named here.
    undermined: list[str] = field(default_factory=list)


@dataclass
class ObserveInput:
    """Input descriptor for observe_many().

    Args:
        content: Raw text of the observed event.
        actors: Named entities involved.
        tags: Categorical labels.
        salience: Subjective importance at encoding time (0-1).
        emotional_valence: Affective weight (-1 to +1).
        timestamp: When the event happened. Defaults to now. Set it when
            loading history, so ``as_of`` places each episode in the period it
            belongs to rather than the moment the import ran.
        provenance: Where the content came from: ``"user"``, ``"tool"`` or
            ``"external"``. Defaults to ``"external"``.
        quarantined: Written after the writer read untrusted content.
    """

    content: str
    actors: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    salience: float = 0.5
    emotional_valence: float = 0.0
    timestamp: datetime | None = None
    provenance: str = DEFAULT_PROVENANCE
    quarantined: bool = False


@dataclass
class CompressionRun:
    """Result of a compress() call."""

    episodes_removed: int
    summaries_created: int
    model_used: str | None = None
    cost_tokens: int = 0


# Convenience type alias used by Store
EmbeddedVector = np.ndarray
