"""Spreading-activation retrieval over the episode-entity graph."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from engram.embedder import Embedder
from engram.models import SearchResult
from engram.store import Store


def spreading_recall(
    query: str,
    k: int,
    store: Store,
    embedder: Embedder,
    depth: int = 2,
    decay: float = 0.5,
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
    as_of: datetime | None = None,
    agent_id: str | None = None,
) -> list[SearchResult]:
    """BFS spreading-activation retrieval.

    Each hop propagates a fraction (``decay * min(edge_weight, 1.0)``) of a
    node's energy to its neighbours. Seeds contribute their cosine score to
    the spread budget, but **not** to the graph-activation score — otherwise
    seeds would be double-counted (once via ``alpha * cosine`` and again via
    ``beta * activation``). Cycles are bounded by the geometric decay across
    the fixed ``depth`` hops.

    Args:
        query: Natural-language search query.
        k: Maximum number of results to return.
        store: Active store instance.
        embedder: Embedder for the query.
        depth: Number of BFS hops from seed nodes.
        decay: Activation multiplier per hop (0-1).
        alpha: Weight of cosine similarity in final score.
        beta: Weight of accumulated graph activation in final score.
        gamma: Weight of importance score in final score.
        as_of: If set, seeds are restricted to episodes with timestamp <= as_of.
        agent_id: If set, restrict to this agent's episodes.

    Returns:
        Top-k :class:`SearchResult` ordered by descending combined score.
    """
    query_vec = embedder.embed(query)
    if as_of is not None:
        seeds = store.search_episodes_as_of(query_vec, k * 3, as_of, agent_id=agent_id)
    else:
        seeds = store.search_episodes(query_vec, k * 3, agent_id=agent_id)

    cosine_scores: dict[str, float] = {ep.id: score for ep, score, _dist in seeds}
    graph_act: dict[str, float] = defaultdict(float)
    spread_source: dict[str, float] = dict(cosine_scores)
    visited_nodes: set[str] = set(cosine_scores.keys())

    # BFS: each frontier node spreads (decay * clamped_weight) of its energy
    # into each neighbour. graph_act accumulates *only* the neighbour-derived
    # signal, so seeds keep graph_act=0 unless something flows back.
    for _hop in range(depth):
        next_source: dict[str, float] = defaultdict(float)
        for node_id, energy in spread_source.items():
            if energy <= 0.0:
                continue
            for neighbor_id, weight in store.get_neighbors(node_id):
                # Hebbian weights can grow >1 from repeated co-occurrence;
                # clamp into [0, 1] so a single hot edge can't dominate.
                clamped = min(max(float(weight), 0.0), 1.0)
                contribution = energy * decay * clamped
                if contribution <= 0.0:
                    continue
                graph_act[neighbor_id] += contribution
                next_source[neighbor_id] += contribution
                visited_nodes.add(neighbor_id)
        if not next_source:
            break
        spread_source = next_source

    # Filter to episode nodes only; apply agent scope when set.
    episodes = store.get_episodes_by_ids(list(visited_nodes), agent_id=agent_id)

    results: list[SearchResult] = []
    for ep in episodes:
        cosine = cosine_scores.get(ep.id, 0.0)
        act = graph_act.get(ep.id, 0.0)
        combined = alpha * cosine + beta * act + gamma * ep.importance_score
        results.append(
            SearchResult(
                episode=ep,
                score=combined,
                distance=0.0,
                importance=ep.importance_score,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    top = results[:k]

    # log_access AFTER sort so the recorded rank reflects what callers actually got.
    now = datetime.now(tz=UTC)
    for rank, result in enumerate(top):
        store.log_access(result.episode.id, now, query, rank)

    return top
