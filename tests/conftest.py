"""Shared test fixtures for embedding benchmark tests."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_chroma(tmp_path):
    """Create a temporary ChromaDB with fake documents and embeddings."""
    import chromadb

    db_path = str(tmp_path / "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("test_papers")

    # Add 6 chunks from 3 papers
    ids = [f"chunk_{i}" for i in range(6)]
    documents = [
        "The Rogers-Ramanujan identities are fundamental in partition theory.",
        "Bailey's lemma provides a mechanism for proving Rogers-Ramanujan type identities.",
        "Crystal bases provide a combinatorial framework for representation theory.",
        "The A2 Bailey lemma extends classical results to higher rank.",
        "Macdonald polynomials generalize several families of symmetric functions.",
        "Hall-Littlewood polynomials are a special case of Macdonald polynomials.",
    ]
    metadatas = [
        {"source": "paper-a.pdf", "category": "core", "chunk_index": 0},
        {"source": "paper-a.pdf", "category": "core", "chunk_index": 1},
        {"source": "paper-b.pdf", "category": "core", "chunk_index": 0},
        {"source": "paper-b.pdf", "category": "core", "chunk_index": 1},
        {"source": "paper-c.pdf", "category": "hall-littlewood", "chunk_index": 0},
        {"source": "paper-c.pdf", "category": "hall-littlewood", "chunk_index": 1},
    ]

    # Create deterministic embeddings (dim=8)
    rng = np.random.RandomState(42)
    embeddings = rng.randn(6, 8).astype(np.float64)
    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    return db_path


@pytest.fixture
def sample_kg(tmp_path):
    """Create a sample knowledge graph JSON file."""
    kg = {
        "metadata": {"version": "1.0"},
        "concepts": [
            {
                "name": "rogers-ramanujan identities",
                "display_name": "Rogers-Ramanujan identities",
                "type": "identity",
                "description": "Famous identities in partition theory.",
                "papers": ["core/paper-a.pdf", "core/paper-b.pdf"],
            },
            {
                "name": "macdonald polynomials",
                "display_name": "Macdonald polynomials",
                "type": "polynomial",
                "description": "Generalized symmetric functions.",
                "papers": [
                    "core/paper-a.pdf",
                    "hall-littlewood/paper-c.pdf",
                    "other/paper-d.pdf",
                ],
            },
            {
                "name": "singleton concept",
                "display_name": "Singleton Concept",
                "type": "theorem",
                "description": "Only linked to one paper.",
                "papers": ["core/paper-b.pdf"],
            },
            {
                "name": "no-match concept",
                "display_name": "No Match Concept",
                "type": "theorem",
                "description": "Papers not in ChromaDB.",
                "papers": ["missing/paper-x.pdf", "missing/paper-y.pdf"],
            },
        ],
        "edges": [],
    }
    kg_path = tmp_path / "knowledge_graph.json"
    kg_path.write_text(json.dumps(kg))
    return str(kg_path)


@pytest.fixture
def sample_source_to_indices():
    """Source→indices mapping matching the sample ChromaDB data."""
    return {
        "paper-a.pdf": [0, 1],
        "paper-b.pdf": [2, 3],
        "paper-c.pdf": [4, 5],
    }
