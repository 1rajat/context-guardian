"""Lazy-loading wrapper around sentence-transformers.

Imports are deferred until the first embed() call so that importing
context_guardian has zero overhead when sentence-transformers is not installed.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: "SentenceTransformer | None" = None


def _get_model() -> "SentenceTransformer":
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic analysis.\n"
                "Install it with:  pip install sentence-transformers"
            )
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(text: str) -> np.ndarray:
    """Embed a single string. Returns shape (384,)."""
    return _get_model().encode(text, convert_to_numpy=True, show_progress_bar=False)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a list of strings. Returns shape (N, 384)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return _get_model().encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
    )


def cosine_similarity_1d(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single vector and each row of a matrix.

    Returns shape (N,) in [-1, 1].
    """
    if matrix.shape[0] == 0:
        return np.array([], dtype=np.float32)
    q = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return (matrix / norms) @ q
