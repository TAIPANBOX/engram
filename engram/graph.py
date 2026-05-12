"""Spreading-activation retrieval over the episode-entity graph."""

from __future__ import annotations

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
) -> list[SearchResult]:
    """BFS spreading-activation retrieval.

    Args:
        query: Natural-language search query.
        k: Maximum number of results to return.
        store: Active store instance.
        embedder: Embedder for the query.
        depth: Number of BFS hops from seed nodes.
        decay: Activation multiplier per hop (0-1).
        alpha: Weight of cosine similarity in final score.
        beta: Weight of graph activation in final score.
        gamma: Weight of importance score in final score.
        as_of: If set, seeds are restricted to episodes with timestamp <= as_of.

    Returns:
        Top-k :class:`SearchResult` ordered by descending combined score.
    """
    query_vec = embedder.embed(query)
    if as_of is not None:
        seeds = store.search_episodes_as_of(query_vec, k * 3, as_of)
    else:
        seeds = store.search_episodes(query_vec, k * 3)

    # Seed activation from cosine similarity.
    activation: dict[str, float] = {}
    cosine_scores: dict[str, float] = {}
    for ep, score, _dist in seeds:
        activation[ep.id] = score
        cosine_scores[ep.id] = score

    # BFS: spread activation outward up to `depth` hops.
    if depth > 0 and activation:
        visited: set[str] = set(activation.keys())
        frontier = list(activation.keys())

        for _hop in range(depth):
            next_frontier: list[str] = []
            for node_id in frontier:
                current_act = activation[node_id]
                for neighbor_id, _weight in store.get_neighbors(node_id):
                    activation[neighbor_id] = activation.get(neighbor_id, 0.0) + current_act * decay
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.append(neighbor_id)
            frontier = next_frontier
            if not frontier:
                break

    # Filter to episode nodes only (entity node ids won't resolve).
    episodes = store.get_episodes_by_ids(list(activation.keys()))

    now = datetime.now(tz=UTC)
    results: list[SearchResult] = []
    for rank, ep in enumerate(episodes):
        cosine = cosine_scores.get(ep.id, 0.0)
        act = activation.get(ep.id, 0.0)
        combined = alpha * cosine + beta * act + gamma * ep.importance_score
        results.append(
            SearchResult(
                episode=ep,
                score=combined,
                distance=0.0,
                importance=ep.importance_score,
            )
        )
        store.log_access(ep.id, now, query, rank)

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]
