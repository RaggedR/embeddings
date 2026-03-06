# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

This is a mixed Rust + Python project built with **maturin**. The Rust crate compiles to a Python extension module.

```bash
# Build the Rust extension and install in the current Python env
maturin develop

# Rust tests (cosine, knn, metrics unit tests)
cargo test

# Python tests (all 33 tests)
python -m pytest tests/ -v

# Single test file or class
python -m pytest tests/test_ground_truth.py -v
python -m pytest tests/test_metrics_integration.py::TestBatchKNN -v

# Run benchmark
python bench.py --models openai-small --k-values 5 10 20 --no-plot
python bench.py --models openai-small openai-large   # compare models
```

After changing any Rust code, you must run `maturin develop` before Python tests or benchmarks will see the changes.

## Architecture

**Purpose**: Benchmark embedding models for mathematical paper retrieval against ground truth from an existing knowledge graph.

**Data source** (external, not in repo): `~/data/arxiv-rag/` — 4,794 paper chunks in ChromaDB (`math_papers` collection), 559 concepts in `knowledge_graph.json`.

### Two-layer design

**Rust layer** (`rust/emb-metrics/src/`) — compute-heavy operations exposed to Python via PyO3:
- `cosine.rs`: Similarity matrices, pairwise distances (rayon-parallelized over queries/rows)
- `knn.rs`: Brute-force batch kNN with partial sort (rayon per query)
- `metrics.rs`: Recall@K, MRR, NDCG@K aggregated over all queries
- `lib.rs`: PyO3 module registration — all functions accept/return numpy arrays via `numpy` crate's zero-copy bindings

**Python layer** (`python/embench/`) — orchestration and I/O:
- `extract.py`: Pulls all chunks + embeddings from ChromaDB, builds `source_to_indices` mapping
- `ground_truth.py`: Maps KG concepts → paper filenames → chunk indices. **Key detail**: KG paper paths have directory prefixes (`core/file.pdf`) while ChromaDB `source` metadata is just the filename (`file.pdf`) — stripping happens via `Path.name`
- `models/base.py`: `EmbeddingModel` ABC — subclasses must return float32, L2-normalized rows from `embed()`
- `models/openai_embed.py`: OpenAI API wrappers. `openai-small` baseline uses pre-extracted ChromaDB embeddings (no API call for corpus)
- `models/mlx_embed.py`: Local models via `mlx-embeddings` (optional dependency)
- `bench.py`: `BenchmarkRunner` orchestrates extract → ground truth → embed → kNN → evaluate
- `report.py`: Comparison tables, JSON export, recall curve plots

**CLI entrypoint**: `bench.py` (root) — argparse wrapper around `BenchmarkRunner`

### Data flow

```
ChromaDB ──extract──→ chunks + embeddings + source_to_indices
                                                    │
KG concepts ──ground_truth──→ query + relevant_indices (108 queries, min_degree=2)
                                                    │
Model.embed(corpus) ──→ corpus_embs (N×D, float32) │
Model.embed(queries) ──→ query_embs (Q×D, float32) │
                              │                     │
              Rust: batch_knn(query_embs, corpus_embs, k) → retrieved indices
              Rust: batch_evaluate(retrieved, relevant, ks) → metrics dict
```

### Ground truth semantics

Each KG concept maps to all chunks from its source papers. This is approximate — not every chunk in a relevant paper directly discusses the concept. The key insight is that this bias is **consistent across models**, so relative comparisons remain valid. Recall@K appears low (~3-4% at K=20) because relevant sets are large (often 100+ chunks). **MRR and NDCG are the meaningful comparison metrics.**

## Fine-tuned model: math-embed

A domain-specific embedding model fine-tuned on the math papers KG. Located at `models/math-embed/final/` (also on HuggingFace as `RobBobin/math-embed`).

**Training**: `train.py` — contrastive learning (MNRL + MatryoshkaLoss) on (concept, chunk) pairs from the KG.
**Training data**: `python/embench/training_data.py` — generates anchor/positive pairs from KG concepts and edges.
**Base model**: `allenai/specter2_base` (SciBERT + citation triplets, 768-dim).

```bash
# Train with default settings (works on Apple Silicon MPS)
python train.py --batch-size 8 --grad-accum 4 --max-seq-length 256 --epochs 3

# Train on different data
python train.py --chroma-path /path/to/chroma --kg-path /path/to/kg.json --output-dir models/new-embed

# Benchmark the result
python bench.py --models openai-small math-embed --no-plot
```

Key files:
- `train.py`: Fine-tuning script with CLI args for all hyperparameters
- `python/embench/training_data.py`: Training data generation from KG + ChromaDB
- `python/embench/models/scientific_embed.py`: SPECTER2, SciNCL, and SentenceTransformerEmbed wrappers
- `paper/math_embeddings.tex`: Research paper documenting the approach

See `CLAUDE_DOCS/embedding-benchmark.md` for the full "repeat for new papers" guide.

## Adding a new embedding model

1. Create a class in `python/embench/models/` extending `EmbeddingModel`
2. Implement `name`, `dim`, and `embed()` (must return float32, L2-normalized)
3. Register it in the `AVAILABLE_MODELS` dict and `make_model()` factory in root `bench.py`

## PyO3/numpy version constraints

Uses pyo3 0.24 + numpy 0.24. These versions require explicit `use numpy::PyUntypedArrayMethods` for `.shape()` and `ndarray::Array2` + `IntoPyArray` for returning 2D arrays (no `PyArray2::from_vec` for 2D).
