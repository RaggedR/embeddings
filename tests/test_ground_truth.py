"""Tests for ground truth construction from knowledge graph."""


class TestBuildGroundTruth:
    def test_basic_construction(self, sample_kg, sample_source_to_indices):
        from embench.ground_truth import build_ground_truth

        gt = build_ground_truth(sample_kg, sample_source_to_indices, min_degree=2)

        # "rogers-ramanujan identities" has 2 matched papers (paper-a, paper-b)
        # "macdonald polynomials" has 2 matched papers (paper-a, paper-c; paper-d missing)
        # "singleton concept" only has 1 paper → excluded
        # "no-match concept" has 0 matched papers → excluded
        assert len(gt) == 2

        names = {entry["concept_name"] for entry in gt}
        assert "rogers-ramanujan identities" in names
        assert "macdonald polynomials" in names

    def test_query_is_display_name(self, sample_kg, sample_source_to_indices):
        from embench.ground_truth import build_ground_truth

        gt = build_ground_truth(sample_kg, sample_source_to_indices, min_degree=2)
        rr = next(e for e in gt if e["concept_name"] == "rogers-ramanujan identities")
        assert rr["query"] == "Rogers-Ramanujan identities"

    def test_relevant_indices_correct(self, sample_kg, sample_source_to_indices):
        from embench.ground_truth import build_ground_truth

        gt = build_ground_truth(sample_kg, sample_source_to_indices, min_degree=2)
        rr = next(e for e in gt if e["concept_name"] == "rogers-ramanujan identities")

        # paper-a has indices [0,1], paper-b has indices [2,3]
        assert set(rr["relevant_indices"]) == {0, 1, 2, 3}

    def test_directory_prefix_stripping(self, sample_kg, sample_source_to_indices):
        from embench.ground_truth import build_ground_truth

        gt = build_ground_truth(sample_kg, sample_source_to_indices, min_degree=2)
        mac = next(e for e in gt if e["concept_name"] == "macdonald polynomials")

        # "core/paper-a.pdf" → paper-a.pdf (indices 0,1)
        # "hall-littlewood/paper-c.pdf" → paper-c.pdf (indices 4,5)
        # "other/paper-d.pdf" → not in ChromaDB, skipped
        assert set(mac["relevant_indices"]) == {0, 1, 4, 5}

    def test_min_degree_filtering(self, sample_kg, sample_source_to_indices):
        from embench.ground_truth import build_ground_truth

        # With min_degree=1, singleton should be included
        gt = build_ground_truth(sample_kg, sample_source_to_indices, min_degree=1)
        names = {e["concept_name"] for e in gt}
        assert "singleton concept" in names

        # With min_degree=3, only macdonald (3 papers listed, but only 2 match)
        gt = build_ground_truth(sample_kg, sample_source_to_indices, min_degree=3)
        assert len(gt) == 0  # None have 3+ matched papers

    def test_no_match_concept_excluded(self, sample_kg, sample_source_to_indices):
        from embench.ground_truth import build_ground_truth

        gt = build_ground_truth(sample_kg, sample_source_to_indices, min_degree=1)
        names = {e["concept_name"] for e in gt}
        assert "no-match concept" not in names
