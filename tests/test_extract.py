"""Tests for ChromaDB extraction."""

import numpy as np


class TestExtractCollection:
    def test_returns_all_fields(self, tmp_chroma):
        from embench.extract import extract_collection

        result = extract_collection(tmp_chroma, "test_papers")

        assert "ids" in result
        assert "documents" in result
        assert "embeddings" in result
        assert "metadatas" in result
        assert "source_to_indices" in result

    def test_correct_count(self, tmp_chroma):
        from embench.extract import extract_collection

        result = extract_collection(tmp_chroma, "test_papers")
        assert len(result["ids"]) == 6
        assert len(result["documents"]) == 6
        assert len(result["metadatas"]) == 6

    def test_embeddings_shape_and_dtype(self, tmp_chroma):
        from embench.extract import extract_collection

        result = extract_collection(tmp_chroma, "test_papers")
        embs = result["embeddings"]
        assert isinstance(embs, np.ndarray)
        assert embs.dtype == np.float32
        assert embs.shape == (6, 8)

    def test_source_to_indices_mapping(self, tmp_chroma):
        from embench.extract import extract_collection

        result = extract_collection(tmp_chroma, "test_papers")
        s2i = result["source_to_indices"]

        assert "paper-a.pdf" in s2i
        assert "paper-b.pdf" in s2i
        assert "paper-c.pdf" in s2i
        assert len(s2i["paper-a.pdf"]) == 2
        assert len(s2i["paper-b.pdf"]) == 2
        assert len(s2i["paper-c.pdf"]) == 2
