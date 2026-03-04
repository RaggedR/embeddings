"""Integration tests for Rust emb_metrics extension from Python."""

import numpy as np
import pytest

from embench import emb_metrics


def _make_normalized(arr):
    """L2-normalize rows of a 2D float32 array."""
    arr = np.array(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / norms


class TestCosineSimMatrix:
    def test_identity_vectors(self):
        vecs = _make_normalized([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = emb_metrics.cosine_similarity_matrix(vecs, vecs)
        np.testing.assert_allclose(result, np.eye(3), atol=1e-5)

    def test_shape(self):
        queries = _make_normalized([[1, 0], [0, 1]])
        corpus = _make_normalized([[1, 0], [0, 1], [1, 1]])
        result = emb_metrics.cosine_similarity_matrix(queries, corpus)
        assert result.shape == (2, 3)


class TestPairwiseDistances:
    def test_diagonal_zero(self):
        vecs = _make_normalized([[1, 0], [0, 1], [1, 1]])
        result = emb_metrics.pairwise_distances(vecs)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-5)

    def test_symmetric(self):
        vecs = _make_normalized([[1, 2], [3, 1], [0.5, 0.5]])
        result = emb_metrics.pairwise_distances(vecs)
        np.testing.assert_allclose(result, result.T, atol=1e-5)


class TestBatchKNN:
    def test_exact_match(self):
        corpus = _make_normalized([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        query = _make_normalized([[0, 1, 0]])
        indices, sims = emb_metrics.batch_knn(query, corpus, 1)
        assert indices[0, 0] == 1
        np.testing.assert_allclose(sims[0, 0], 1.0, atol=1e-5)

    def test_output_shapes(self):
        corpus = _make_normalized(np.random.randn(10, 4))
        queries = _make_normalized(np.random.randn(3, 4))
        indices, sims = emb_metrics.batch_knn(queries, corpus, 5)
        assert indices.shape == (3, 5)
        assert sims.shape == (3, 5)

    def test_descending_similarity(self):
        corpus = _make_normalized(np.random.randn(20, 4))
        query = _make_normalized(np.random.randn(1, 4))
        _, sims = emb_metrics.batch_knn(query, corpus, 10)
        # Similarities should be in descending order
        for i in range(9):
            assert sims[0, i] >= sims[0, i + 1]


class TestBatchEvaluate:
    def test_perfect_recall(self):
        retrieved = [[0, 1, 2, 3, 4]]
        relevant = [[0, 1, 2]]
        metrics = emb_metrics.batch_evaluate(retrieved, relevant, [5])
        assert abs(metrics["recall@5"] - 1.0) < 1e-10

    def test_mrr(self):
        # First relevant at rank 3 (0-indexed: position 2)
        retrieved = [[10, 11, 0, 1, 2]]
        relevant = [[0]]
        metrics = emb_metrics.batch_evaluate(retrieved, relevant, [5])
        assert abs(metrics["mrr"] - 1.0 / 3.0) < 1e-10

    def test_consistency_with_knn(self):
        """Verify kNN + batch_evaluate pipeline works end-to-end."""
        rng = np.random.RandomState(42)
        corpus = _make_normalized(rng.randn(50, 8))
        queries = _make_normalized(rng.randn(5, 8))

        indices, sims = emb_metrics.batch_knn(queries, corpus, 10)

        retrieved = [row.tolist() for row in indices]
        # Mock ground truth: first 5 indices are always relevant
        relevant = [[0, 1, 2, 3, 4]] * 5

        metrics = emb_metrics.batch_evaluate(retrieved, relevant, [5, 10])
        assert "recall@5" in metrics
        assert "recall@10" in metrics
        assert "mrr" in metrics
        assert "ndcg@5" in metrics
        assert 0.0 <= metrics["recall@10"] <= 1.0
