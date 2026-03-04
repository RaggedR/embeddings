"""Tests for benchmark orchestration and reporting."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embench.bench import ModelResult
from embench.report import print_comparison_table, save_json


class TestModelResult:
    def test_dataclass_creation(self):
        r = ModelResult(
            model_name="test",
            dim=128,
            metrics={"recall@5": 0.5, "mrr": 0.3},
            embed_time_s=1.0,
            eval_time_s=0.1,
            k_values=[5, 10],
        )
        assert r.model_name == "test"
        assert r.metrics["recall@5"] == 0.5


class TestReport:
    def test_print_comparison_table_no_results(self, capsys):
        print_comparison_table([])
        captured = capsys.readouterr()
        assert "No results" in captured.out

    def test_print_comparison_table(self, capsys):
        results = [
            ModelResult("model-a", 128, {"recall@5": 0.5, "recall@10": 0.7, "mrr": 0.3, "ndcg@10": 0.4},
                        1.0, 0.1, [5, 10]),
            ModelResult("model-b", 256, {"recall@5": 0.6, "recall@10": 0.8, "mrr": 0.4, "ndcg@10": 0.5},
                        2.0, 0.2, [5, 10]),
        ]
        print_comparison_table(results)
        captured = capsys.readouterr()
        assert "model-a" in captured.out
        assert "model-b" in captured.out
        assert "0.5000" in captured.out

    def test_save_json(self, tmp_path):
        results = [
            ModelResult("test-model", 128, {"recall@5": 0.5}, 1.0, 0.1, [5]),
        ]
        path = save_json(results, str(tmp_path))
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["models"][0]["name"] == "test-model"
        assert data["models"][0]["metrics"]["recall@5"] == 0.5
