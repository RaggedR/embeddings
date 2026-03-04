"""Benchmark orchestration."""

import time
from dataclasses import dataclass, field

import numpy as np

from embench.models.base import EmbeddingModel


@dataclass
class ModelResult:
    """Results for a single model."""

    model_name: str
    dim: int
    metrics: dict[str, float]
    embed_time_s: float
    eval_time_s: float
    k_values: list[int]


@dataclass
class BenchmarkRunner:
    """Orchestrates embedding benchmark across models."""

    chroma_path: str
    kg_path: str
    output_dir: str = "results"
    collection_name: str = "math_papers"
    min_degree: int = 2
    k_values: list[int] = field(default_factory=lambda: [5, 10, 20])

    _data: dict | None = field(default=None, init=False, repr=False)
    _ground_truth: list[dict] | None = field(default=None, init=False, repr=False)

    def _ensure_data(self):
        """Load ChromaDB data and build ground truth (cached)."""
        if self._data is None:
            from embench.extract import extract_collection
            from embench.ground_truth import build_ground_truth

            print("Extracting ChromaDB collection...")
            self._data = extract_collection(self.chroma_path, self.collection_name)
            print(f"  {len(self._data['ids'])} chunks, dim={self._data['embeddings'].shape[1]}")

            print("Building ground truth from knowledge graph...")
            self._ground_truth = build_ground_truth(
                self.kg_path,
                self._data["source_to_indices"],
                min_degree=self.min_degree,
            )
            print(f"  {len(self._ground_truth)} queries")

    def run_model(
        self,
        model: EmbeddingModel,
        corpus_embeddings: np.ndarray | None = None,
    ) -> ModelResult:
        """Benchmark a single model.

        Args:
            model: The embedding model to evaluate.
            corpus_embeddings: Pre-computed corpus embeddings (e.g., from ChromaDB for baseline).
                If None, embeds the full corpus using the model.

        Returns:
            ModelResult with metrics and timing.
        """
        self._ensure_data()

        queries = [gt["query"] for gt in self._ground_truth]
        relevant = [gt["relevant_indices"] for gt in self._ground_truth]

        # Embed corpus
        if corpus_embeddings is not None:
            print(f"  Using pre-computed corpus embeddings ({corpus_embeddings.shape})")
            corpus_embs = corpus_embeddings.astype(np.float32)
        else:
            print(f"  Embedding {len(self._data['documents'])} corpus chunks...")
            t0 = time.time()
            corpus_embs = model.embed(self._data["documents"])
            embed_corpus_time = time.time() - t0
            print(f"  Corpus embedded in {embed_corpus_time:.1f}s")

        # Embed queries
        print(f"  Embedding {len(queries)} queries...")
        t0 = time.time()
        query_embs = model.embed(queries)
        embed_time = time.time() - t0

        if corpus_embeddings is not None:
            embed_time = 0.0  # Don't count pre-computed

        # kNN search via Rust
        from embench import emb_metrics

        max_k = max(self.k_values)
        t0 = time.time()
        indices, _sims = emb_metrics.batch_knn(query_embs, corpus_embs, max_k)
        eval_time = time.time() - t0

        # Convert kNN results to list-of-lists for batch_evaluate
        retrieved = [row.tolist() for row in indices]

        # Compute metrics via Rust
        metrics = emb_metrics.batch_evaluate(retrieved, relevant, self.k_values)

        print(f"  {model.name}: recall@{max_k}={metrics.get(f'recall@{max_k}', 0):.3f}, "
              f"mrr={metrics.get('mrr', 0):.3f}")

        return ModelResult(
            model_name=model.name,
            dim=model.dim,
            metrics=metrics,
            embed_time_s=embed_time,
            eval_time_s=eval_time,
            k_values=self.k_values,
        )

    def run_all(self, models: list[tuple[EmbeddingModel, np.ndarray | None]]) -> list[ModelResult]:
        """Run benchmark for all models sequentially.

        Args:
            models: List of (model, optional_precomputed_embeddings) tuples.

        Returns:
            List of ModelResult.
        """
        self._ensure_data()
        results = []

        for model, precomputed in models:
            print(f"\n--- {model.name} ---")
            result = self.run_model(model, corpus_embeddings=precomputed)
            results.append(result)

        return results
