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
    from engram.llm import LLMAdapter


def reflect(
    store: Store,
    llm: LLMAdapter | None,
    cfg: DecayConfig,
    now: datetime | None = None,
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
        try:
            raw_facts, call_tokens = llm.extract_facts(episodes)
        except Exception:
            # Roll the run record back so it doesn't linger unfinished and
            # poison the incremental window on the next reflection.
            store.delete_reflection(run_id)
            raise
        cost_tokens += call_tokens
        episode_ids = [ep.id for ep in episodes]

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

            # Contradiction detection: close any older active facts with same (s, p).
            conflicts = store.get_active_facts(fact.subject, fact.predicate)
            for old in conflicts:
                if old.id != fact.id:
                    store.close_fact(old.id, valid_to=now, superseded_by=fact.id)
                    contradictions_resolved += 1

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
