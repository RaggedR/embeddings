# Feature: Embedding Benchmark
> Benchmarks multiple embedding models for mathematical paper retrieval using ground truth from a knowledge graph.

## Overview
INSTINCT uses `text-embedding-3-small` (OpenAI, 1536-dim) for all retrieval. This project measures whether
better embedding models improve retrieval of mathematical content. Ground truth is derived from the knowledge
graph in `~/data/arxiv-rag/knowledge_graph.json` — each concept maps to its source papers, whose chunks are
indexed in ChromaDB.

The system combines a Rust extension (via PyO3/maturin) for fast similarity/kNN/metric computation with
Python orchestration for model loading, ChromaDB extraction, and reporting.

## Resources
- Data: `~/data/arxiv-rag/` (4,794 chunks in ChromaDB, 559 concepts in KG)
- Parent project: INSTINCT (`~/git/math-research-tools/`)

## Assets
- Rust crate: `rust/emb-metrics/src/`
- Python package: `python/embench/`
- CLI entrypoint: `bench.py`
- Tests: `tests/`
