"""Reflection loop: extract facts, detect contradictions, decay, prune."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from engram.decay import run_decay
from engram.importance import DecayConfig
from engram.llm import MAX_EXTRACTED_CONFIDENCE
from engram.models import Fact, ReflectionRun
from engram.store import Store

if TYPE_CHECKING:
    from engram.events import EventLog
    from engram.llm import LLMAdapter


def reflect(
    store: Store,
    llm: LLMAdapter | None,
    cfg: DecayConfig,
    now: datetime | None = None,
    events: EventLog | None = None,
    agent_id: str | None = None,
) -> ReflectionRun:
    """Run one reflection pass.

    Steps:
    1. Record a new ReflectionRun.
    2. Fetch unreflected episodes (since the last run).
    3. If an LLM is configured: extract facts, detect and resolve contradictions.
    4. Recompute importance scores (decay).
    5. Prune episodes below the importance threshold.
    6. Finalise the ReflectionRun record.

    Args:
        store: Active store instance.
        llm: LLM adapter for fact extraction, or None to skip extraction.
        cfg: Decay and pruning configuration.
        now: Reference time (defaults to current UTC time).
        events: Optional Agent Passport event log. When given, a
            ``contradiction_found`` event is emitted at the point a
            newly-extracted fact supersedes an older one with a *different*
            object; a same-object re-extraction supersedes silently, as
            agreement (see :mod:`engram.events`). ``None`` (the default)
            emits nothing.
        agent_id: Agent scope to attach to emitted events. Ignored when
            *events* is ``None``.

    Returns:
        The completed :class:`ReflectionRun`.
    """
    if now is None:
        now = datetime.now(tz=UTC)

    # Window off the last *completed* run: an aborted run must not advance
    # `since` and skip the episodes it never managed to process.
    previous_run = store.get_last_finished_reflection()
    since: datetime | None = previous_run.started_at if previous_run else None

    run_id = str(uuid.uuid4())
    model_name = llm.model_name if llm else None
    run = ReflectionRun(id=run_id, started_at=now, model_used=model_name)
    store.insert_reflection(run)

    episodes = store.get_episodes_since(since)
    facts_extracted = 0
    contradictions_resolved = 0

    cost_tokens = 0
    if llm and episodes:
        # The whole extract -> build-facts -> insert -> contradiction -> edges
        # sequence is protected as one unit: a mid-way raise anywhere in here
        # (a malformed fact dict, a Store error, anything) must not leave
        # partial output behind, or it gets duplicated when the next
        # reflection reprocesses this same episode window.
        try:
            raw_facts, call_tokens = llm.extract_facts(episodes)
            cost_tokens += call_tokens
            episode_ids = [ep.id for ep in episodes]

            # store.transaction() wraps only the DB-mutating part below, not
            # the LLM call above: the lock it holds must never span a slow
            # external call (see _synchronized's docstring in store.py). A
            # raise inside this block rolls back every fact/edge/
            # contradiction write it made, in one shot -- including partial
            # Hebbian edge-weight upserts that a manual delete could not
            # safely reverse (see Store.transaction()).
            with store.transaction():
                for rf in raw_facts:
                    # Belt-and-braces: cap confidence again in case the adapter
                    # bypasses _parse_facts_json (e.g. test stubs feeding raw dicts).
                    confidence = min(
                        max(float(rf.get("confidence", 0.5)), 0.0),
                        MAX_EXTRACTED_CONFIDENCE,
                    )
                    fact = Fact(
                        id=str(uuid.uuid4()),
                        subject=str(rf["subject"]),
                        predicate=str(rf["predicate"]),
                        object=str(rf["object"]),
                        valid_from=now,
                        valid_to=None,
                        recorded_at=now,
                        superseded_at=None,
                        superseded_by=None,
                        confidence=confidence,
                        derived_from=episode_ids,
                        extracted_by=run_id,
                    )
                    store.insert_fact(fact)
                    facts_extracted += 1

                    # Supersede any older active facts with the same (s, p):
                    # the newest extraction wins, leaving one active value per
                    # pair. Only a differing object is a contradiction (counted
                    # and emitted); re-extracting the same object is agreement,
                    # so the supersede chain is refreshed silently -- mirroring
                    # contradictions(), which skips same-object pairs too.
                    conflicts = store.get_active_facts(fact.subject, fact.predicate)
                    for old in conflicts:
                        if old.id == fact.id:
                            continue
                        store.close_fact(old.id, valid_to=now, superseded_by=fact.id)
                        if old.object == fact.object:
                            continue
                        contradictions_resolved += 1
                        if events is not None:
                            events.emit(
                                "contradiction_found",
                                agent_id,
                                {"memory_id": fact.id, "conflicting_memory_id": old.id},
                                run_id=run_id,
                            )

                    # Entity extraction and episode→entity edges (Hebbian reinforcement).
                    for name in (fact.subject, fact.object):
                        entity = store.find_or_create_entity(name, "unknown", now)
                        for ep_id in episode_ids:
                            store.insert_edge(
                                ep_id,
                                entity.id,
                                "mentions",
                                weight=fact.confidence,
                                created_at=now,
                            )
        except Exception:
            # By this point store.transaction() (if it was entered at all)
            # has already rolled back every fact/edge/contradiction write
            # from this run. Only the ReflectionRun row itself -- inserted,
            # and committed, before this block started -- still needs its
            # own explicit cleanup so it doesn't linger unfinished and
            # poison the incremental window on the next reflection.
            store.delete_reflection(run_id)
            raise

    # Decay + prune.
    run_decay(store, cfg, now)
    store.prune_episodes(cfg.threshold)

    store.update_reflection(
        run_id,
        finished_at=now,
        episodes_processed=len(episodes),
        facts_extracted=facts_extracted,
        contradictions_resolved=contradictions_resolved,
        cost_tokens=cost_tokens,
    )
    run.finished_at = now
    run.episodes_processed = len(episodes)
    run.facts_extracted = facts_extracted
    run.contradictions_resolved = contradictions_resolved
    run.cost_tokens = cost_tokens
    return run
