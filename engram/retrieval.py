"""Retrieval logic: maps a query string to a ranked list of SearchResults."""

from __future__ import annotations

from engram.embedder import Embedder
from engram.models import SearchResult
from engram.store import Store


def recall(query: str, k: int, store: Store, embedder: Embedder) -> list[SearchResult]:
    """Return the top-k episodes most similar to *query*.

    Uses cosine-equivalent ranking via sqlite-vec's L2 index.
    """
    query_vec = embedder.embed(query)
    hits = store.search_episodes(query_vec, k)
    return [SearchResult(episode=ep, score=score, distance=dist) for ep, score, dist in hits]
