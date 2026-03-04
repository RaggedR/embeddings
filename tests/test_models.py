"""Tests for embedding model interfaces."""

import numpy as np
import pytest

from embench.models.base import EmbeddingModel


class DummyModel(EmbeddingModel):
    """Minimal concrete implementation for testing the ABC."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def dim(self) -> int:
        return 4

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        # Return deterministic embeddings based on text length
        rng = np.random.RandomState(42)
        arr = rng.randn(len(texts), self.dim).astype(np.float32)
        return self.l2_normalize(arr)


class TestEmbeddingModelABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            EmbeddingModel()

    def test_concrete_implementation(self):
        model = DummyModel()
        assert model.name == "dummy"
        assert model.dim == 4

    def test_embed_returns_normalized(self):
        model = DummyModel()
        embs = model.embed(["hello", "world"])
        assert embs.shape == (2, 4)
        assert embs.dtype == np.float32
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_embed_query_returns_1d(self):
        model = DummyModel()
        emb = model.embed_query("hello")
        assert emb.shape == (4,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-6

    def test_l2_normalize_handles_zero_vector(self):
        zero = np.zeros(4, dtype=np.float32)
        result = EmbeddingModel.l2_normalize(zero)
        np.testing.assert_array_equal(result, zero)

    def test_l2_normalize_2d(self):
        arr = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        result = EmbeddingModel.l2_normalize(arr)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)
        np.testing.assert_allclose(result[0], [0.6, 0.8], atol=1e-6)


class TestOpenAIModels:
    """Test OpenAI model configuration (not API calls)."""

    def test_openai_small_config(self):
        from embench.models.openai_embed import OpenAISmall
        model = OpenAISmall()
        assert model.name == "openai-small"
        assert model.dim == 1536

    def test_openai_large_config(self):
        from embench.models.openai_embed import OpenAILarge
        model = OpenAILarge()
        assert model.name == "openai-large"
        assert model.dim == 3072


class TestMLXModels:
    """Test MLX model configuration (skipped if mlx_embeddings not installed)."""

    def test_bge_large_requires_mlx(self):
        try:
            import mlx_embeddings  # noqa: F401
            pytest.skip("mlx_embeddings is installed, can't test import error")
        except ImportError:
            from embench.models.mlx_embed import BGELarge
            with pytest.raises(ImportError, match="mlx-embeddings"):
                BGELarge()
