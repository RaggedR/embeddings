"""Extract chunks and embeddings from ChromaDB."""

from collections import defaultdict

import numpy as np


def extract_collection(
    chroma_path: str,
    collection_name: str = "math_papers",
) -> dict:
    """Extract all data from a ChromaDB collection.

    Returns:
        dict with keys:
            ids: list[str]
            documents: list[str]
            embeddings: np.ndarray (N×D, float32)
            metadatas: list[dict]
            source_to_indices: dict[str, list[int]]
    """
    import chromadb

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(collection_name)

    data = collection.get(include=["documents", "embeddings", "metadatas"])

    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    # Convert embeddings to float32 numpy array
    embeddings = np.array(data["embeddings"], dtype=np.float32)

    # Build source → chunk indices mapping
    source_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, meta in enumerate(metadatas):
        source = meta.get("source", "")
        if source:
            source_to_indices[source].append(i)

    return {
        "ids": ids,
        "documents": documents,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "source_to_indices": dict(source_to_indices),
    }
