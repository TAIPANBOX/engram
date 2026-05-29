"""Local embedding model wrapper via fastembed."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fastembed import TextEmbedding as _TextEmbedding

DEFAULT_MODEL: str = "BAAI/bge-small-en-v1.5"
_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
}

_DEFAULT_CACHE_SIZE = 256


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return vec / ||vec||₂, or vec unchanged if its norm is ~0.

    Forcing unit length means sqlite-vec's L2 distance becomes a monotone
    function of cosine similarity (L2^2 = 2 - 2*cos for unit vectors), so the
    "cosine" score is correct for any embedder, not just those that happen
    to normalize internally.
    """
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec
    return vec / norm


class Embedder:
    """Wraps a fastembed TextEmbedding model for local, dependency-light inference.

    Maintains an LRU cache of up to *cache_size* vectors. Repeated queries
    (common in agent recall loops) return the cached vector without re-running
    ONNX inference.

    All returned vectors are L2-normalized so downstream similarity scoring
    is metric-correct regardless of which fastembed model is selected.
    """

    def __init__(
        self, model_name: str = DEFAULT_MODEL, cache_size: int = _DEFAULT_CACHE_SIZE
    ) -> None:
        self._model_name = model_name
        self._model: _TextEmbedding | None = None
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._dim: int | None = _KNOWN_DIMS.get(model_name)

    def _get_model(self) -> _TextEmbedding:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self._model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string. Returns a unit-length float32 ndarray of shape (dim,).

        The result is served from the LRU cache when the same text has been
        embedded before, avoiding ONNX inference overhead on repeated queries.
        """
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached
        result = next(iter(self._get_model().embed([text])))
        vec = _l2_normalize(np.array(result, dtype=np.float32))
        if self._dim is None:
            self._dim = int(vec.shape[0])
        self._cache[text] = vec
        self._cache.move_to_end(text)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple strings using a single ONNX inference pass.

        Texts already in the cache are returned directly; only uncached texts
        are sent through the model. Returns unit-length results in the same
        order as *texts*.
        """
        if not texts:
            return []

        # Collect results in order, run inference only for cache misses.
        result: list[np.ndarray | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for i, t in enumerate(texts):
            cached = self._cache.get(t)
            if cached is not None:
                self._cache.move_to_end(t)
                result[i] = cached
            else:
                missing_indices.append(i)
                missing_texts.append(t)

        if missing_texts:
            new_vecs = [
                _l2_normalize(np.array(v, dtype=np.float32))
                for v in self._get_model().embed(missing_texts)
            ]
            for i, text, vec in zip(missing_indices, missing_texts, new_vecs, strict=True):
                if self._dim is None:
                    self._dim = int(vec.shape[0])
                result[i] = vec
                self._cache[text] = vec
                self._cache.move_to_end(text)
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)

        return result  # type: ignore[return-value]  # all slots filled above

    @property
    def dim(self) -> int:
        """Output dimensionality of this model.

        For known fastembed models the value is taken from a static table; for
        any other model we run a one-token inference to probe the true
        dimensionality rather than silently defaulting to 384 (which would
        corrupt the vec0 table for a 768- or 1024-dim model).
        """
        if self._dim is None:
            self.embed("")
            assert self._dim is not None
        return self._dim
