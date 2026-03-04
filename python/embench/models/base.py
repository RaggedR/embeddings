"""Abstract base class for embedding models."""

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    """Interface for embedding models used in benchmarking."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""

    @abstractmethod
    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: Documents to embed.
            batch_size: Number of texts per batch.

        Returns:
            np.ndarray of shape (len(texts), self.dim), dtype float32, L2-normalized rows.
        """

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Returns:
            np.ndarray of shape (self.dim,), dtype float32, L2-normalized.
        """
        return self.embed([query])[0]

    @staticmethod
    def l2_normalize(arr: np.ndarray) -> np.ndarray:
        """L2-normalize rows of a 2D array (or a 1D vector)."""
        if arr.ndim == 1:
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 0 else arr
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return arr / norms
