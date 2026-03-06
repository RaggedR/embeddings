"""Build training dataset from knowledge graph + ChromaDB for contrastive fine-tuning.

Generates (anchor, positive) pairs where:
  - anchor = concept display_name or description
  - positive = a chunk from one of the concept's source papers

For MultipleNegativesRankingLoss, in-batch negatives serve as hard negatives
automatically — no explicit negative mining needed.
"""

import json
import random
from pathlib import Path

from embench.extract import extract_collection


def build_training_pairs(
    kg_path: str,
    chroma_path: str,
    max_positives_per_concept: int = 20,
    include_descriptions: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Build (anchor, positive) training pairs from KG + ChromaDB.

    For each concept, pairs the concept name (and optionally description)
    with chunks from its source papers.

    Args:
        kg_path: Path to knowledge_graph.json
        chroma_path: Path to ChromaDB directory
        max_positives_per_concept: Cap positives per concept to avoid
            over-representing concepts with many papers
        include_descriptions: Also use concept descriptions as anchors
            (doubles the number of anchor variants)
        seed: Random seed for reproducible sampling

    Returns:
        List of dicts with keys: anchor, positive, concept_name
    """
    rng = random.Random(seed)

    # Load data
    with open(kg_path) as f:
        kg = json.load(f)

    data = extract_collection(chroma_path)
    documents = data["documents"]
    source_to_indices = data["source_to_indices"]

    pairs = []

    for concept in kg["concepts"]:
        papers = concept.get("papers", [])
        if not papers:
            continue

        # Collect all chunk indices for this concept's papers
        chunk_indices = []
        for paper_path in papers:
            filename = Path(paper_path).name
            if filename in source_to_indices:
                chunk_indices.extend(source_to_indices[filename])

        if not chunk_indices:
            continue

        # Sample chunks if too many
        if len(chunk_indices) > max_positives_per_concept:
            chunk_indices = rng.sample(chunk_indices, max_positives_per_concept)

        # Pair concept name with each chunk
        display_name = concept["display_name"]
        for idx in chunk_indices:
            pairs.append({
                "anchor": display_name,
                "positive": documents[idx],
                "concept_name": concept["name"],
            })

        # Also pair concept description with each chunk
        if include_descriptions and concept.get("description"):
            description = concept["description"]
            for idx in chunk_indices:
                pairs.append({
                    "anchor": description,
                    "positive": documents[idx],
                    "concept_name": concept["name"],
                })

    rng.shuffle(pairs)
    return pairs


def build_edge_pairs(
    kg_path: str,
    chroma_path: str,
    max_per_edge: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Build cross-concept pairs from KG edges.

    If concept A 'generalizes' concept B, then chunks from A's papers
    are soft-positives for concept B's name, and vice versa. These
    teach the model about mathematical relationships between concepts.

    Args:
        kg_path: Path to knowledge_graph.json
        chroma_path: Path to ChromaDB directory
        max_per_edge: Max pairs per edge direction
        seed: Random seed

    Returns:
        List of dicts with keys: anchor, positive, relation
    """
    rng = random.Random(seed)

    with open(kg_path) as f:
        kg = json.load(f)

    data = extract_collection(chroma_path)
    documents = data["documents"]
    source_to_indices = data["source_to_indices"]

    # Build concept name → concept lookup
    concept_map = {c["name"]: c for c in kg["concepts"]}

    def get_chunks(concept_name: str) -> list[int]:
        concept = concept_map.get(concept_name)
        if not concept:
            return []
        indices = []
        for paper_path in concept.get("papers", []):
            filename = Path(paper_path).name
            if filename in source_to_indices:
                indices.extend(source_to_indices[filename])
        return indices

    pairs = []

    for edge in kg["edges"]:
        source_name = edge["source"]
        target_name = edge["target"]
        relation = edge["relation"]

        source_concept = concept_map.get(source_name)
        target_concept = concept_map.get(target_name)
        if not source_concept or not target_concept:
            continue

        # Source concept name → target's chunks
        target_chunks = get_chunks(target_name)
        if target_chunks:
            sampled = rng.sample(target_chunks, min(max_per_edge, len(target_chunks)))
            for idx in sampled:
                pairs.append({
                    "anchor": source_concept["display_name"],
                    "positive": documents[idx],
                    "relation": relation,
                })

        # Target concept name → source's chunks
        source_chunks = get_chunks(source_name)
        if source_chunks:
            sampled = rng.sample(source_chunks, min(max_per_edge, len(source_chunks)))
            for idx in sampled:
                pairs.append({
                    "anchor": target_concept["display_name"],
                    "positive": documents[idx],
                    "relation": relation,
                })

    rng.shuffle(pairs)
    return pairs


def build_dataset(
    kg_path: str,
    chroma_path: str,
    include_edges: bool = True,
    max_positives_per_concept: int = 20,
    max_per_edge: int = 5,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """Build complete training dataset with train/val split.

    Returns:
        Dict with keys:
            train: list of {anchor, positive} dicts
            val: list of {anchor, positive} dicts
            stats: summary statistics
    """
    rng = random.Random(seed)

    # Direct concept-chunk pairs
    direct_pairs = build_training_pairs(
        kg_path, chroma_path,
        max_positives_per_concept=max_positives_per_concept,
        include_descriptions=True,
        seed=seed,
    )

    # Cross-concept edge pairs
    edge_pairs = []
    if include_edges:
        edge_pairs = build_edge_pairs(
            kg_path, chroma_path,
            max_per_edge=max_per_edge,
            seed=seed,
        )

    # Combine and deduplicate
    all_pairs = []
    seen = set()
    for p in direct_pairs + edge_pairs:
        key = (p["anchor"][:50], p["positive"][:50])
        if key not in seen:
            seen.add(key)
            all_pairs.append({"anchor": p["anchor"], "positive": p["positive"]})

    rng.shuffle(all_pairs)

    # Train/val split
    n_val = max(1, int(len(all_pairs) * val_fraction))
    val_set = all_pairs[:n_val]
    train_set = all_pairs[n_val:]

    # Count unique anchors and concepts
    unique_anchors = len(set(p["anchor"] for p in all_pairs))

    stats = {
        "total_pairs": len(all_pairs),
        "direct_pairs": len(direct_pairs),
        "edge_pairs": len(edge_pairs),
        "train_pairs": len(train_set),
        "val_pairs": len(val_set),
        "unique_anchors": unique_anchors,
    }

    return {"train": train_set, "val": val_set, "stats": stats}
