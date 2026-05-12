"""Local embedding model wrapper via fastembed."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fastembed import TextEmbedding as _TextEmbedding

DEFAULT_MODEL: str = "BAAI/bge-small-en-v1.5"
_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
}


class Embedder:
    """Wraps a fastembed TextEmbedding model for local, dependency-light inference."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: _TextEmbedding | None = None

    def _get_model(self) -> _TextEmbedding:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self._model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string. Returns a float32 ndarray of shape (dim,)."""
        result = list(self._get_model().embed([text]))[0]
        return np.array(result, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple strings. Returns a list of float32 ndarrays."""
        return [np.array(v, dtype=np.float32) for v in self._get_model().embed(texts)]

    @property
    def dim(self) -> int:
        """Output dimensionality of this model."""
        return _KNOWN_DIMS.get(self._model_name, 384)
