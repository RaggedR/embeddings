"""Build ground truth from knowledge graph for retrieval evaluation."""

import json
from pathlib import Path


def build_ground_truth(
    kg_path: str,
    source_to_indices: dict[str, list[int]],
    min_degree: int = 2,
) -> list[dict]:
    """Build ground truth queries from knowledge graph.

    For each concept with >= min_degree papers that have chunks in ChromaDB:
    - query = concept display_name
    - relevant_indices = union of chunk indices for all the concept's papers
    - concept_name = concept name

    Paper paths in KG have directory prefixes (e.g., "core/file.pdf").
    ChromaDB source metadata is just the filename (e.g., "file.pdf").
    We strip the directory prefix when matching.

    Args:
        kg_path: Path to knowledge_graph.json
        source_to_indices: Maps source filename → list of chunk indices (from ChromaDB)
        min_degree: Minimum number of matched papers required per concept

    Returns:
        List of dicts with keys: query, relevant_indices, concept_name
    """
    with open(kg_path) as f:
        kg = json.load(f)

    ground_truth = []

    for concept in kg["concepts"]:
        papers = concept.get("papers", [])
        if not papers:
            continue

        # Collect chunk indices for all papers in this concept
        relevant_indices = []
        matched_papers = 0

        for paper_path in papers:
            # Strip directory prefix: "core/file.pdf" → "file.pdf"
            filename = Path(paper_path).name
            if filename in source_to_indices:
                relevant_indices.extend(source_to_indices[filename])
                matched_papers += 1

        if matched_papers >= min_degree and relevant_indices:
            ground_truth.append({
                "query": concept["display_name"],
                "relevant_indices": relevant_indices,
                "concept_name": concept["name"],
            })

    return ground_truth
