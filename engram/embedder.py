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


class Embedder:
    """Wraps a fastembed TextEmbedding model for local, dependency-light inference.

    Maintains an LRU cache of up to *cache_size* vectors. Repeated queries
    (common in agent recall loops) return the cached vector without re-running
    ONNX inference.
    """

    def __init__(
        self, model_name: str = DEFAULT_MODEL, cache_size: int = _DEFAULT_CACHE_SIZE
    ) -> None:
        self._model_name = model_name
        self._model: _TextEmbedding | None = None
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def _get_model(self) -> _TextEmbedding:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self._model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string. Returns a float32 ndarray of shape (dim,).

        The result is served from the LRU cache when the same text has been
        embedded before, avoiding ONNX inference overhead on repeated queries.
        """
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached
        result = next(iter(self._get_model().embed([text])))
        vec = np.array(result, dtype=np.float32)
        self._cache[text] = vec
        self._cache.move_to_end(text)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple strings using a single ONNX inference pass.

        Texts already in the cache are returned directly; only uncached texts
        are sent through the model. Returns results in the same order as *texts*.
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
                np.array(v, dtype=np.float32) for v in self._get_model().embed(missing_texts)
            ]
            for i, text, vec in zip(missing_indices, missing_texts, new_vecs, strict=True):
                result[i] = vec
                self._cache[text] = vec
                self._cache.move_to_end(text)
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)

        return result  # type: ignore[return-value]  # all slots filled above

    @property
    def dim(self) -> int:
        """Output dimensionality of this model."""
        return _KNOWN_DIMS.get(self._model_name, 384)
