"""MLX-based local embedding models (Apple Silicon only)."""

import numpy as np

from embench.models.base import EmbeddingModel


def _check_mlx():
    """Check if mlx_embeddings is available."""
    try:
        import mlx_embeddings  # noqa: F401
        return True
    except ImportError:
        return False


class MLXEmbed(EmbeddingModel):
    """Wrapper for models loaded via mlx-embeddings."""

    def __init__(self, model_repo: str, model_name: str, dimensions: int):
        if not _check_mlx():
            raise ImportError(
                "mlx-embeddings is required for local models. "
                "Install with: pip install mlx-embeddings"
            )
        from mlx_embeddings import load

        self._name = model_name
        self._dim = dimensions
        self._model, self._tokenizer = load(model_repo)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        from mlx_embeddings import encode

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = encode(self._model, self._tokenizer, batch)
            # mlx arrays → numpy
            all_embeddings.append(np.array(embs, dtype=np.float32))
        arr = np.concatenate(all_embeddings, axis=0)
        return self.l2_normalize(arr)


class BGELarge(MLXEmbed):
    """BAAI/bge-large-en-v1.5 via MLX (1024-dim)."""

    def __init__(self):
        super().__init__(
            "BAAI/bge-large-en-v1.5",
            "bge-large",
            1024,
        )


class Qwen3Embed(MLXEmbed):
    """Qwen/Qwen3-Embedding-0.6B via MLX (1024-dim)."""

    def __init__(self):
        super().__init__(
            "Qwen/Qwen3-Embedding-0.6B",
            "qwen3-0.6b",
            1024,
        )
