# embeddings

Benchmark embedding models for mathematical paper retrieval against ground truth from a knowledge graph.

Includes **[math-embed](https://huggingface.co/RobBobin/math-embed)** — a domain-specific embedding model fine-tuned on combinatorics papers using knowledge-graph-guided contrastive learning.

## Results

Benchmarked on mathematical paper retrieval (108 queries, 4,794 paper chunks):

| Model | MRR | NDCG@10 |
|-------|-----|---------|
| **[math-embed](https://huggingface.co/RobBobin/math-embed)** | **0.816** | **0.736** |
| OpenAI text-embedding-3-small | 0.461 | 0.324 |
| SPECTER2 (proximity adapter) | 0.360 | 0.225 |
| SciNCL | 0.306 | 0.205 |

## Architecture

**Rust layer** (PyO3) — compute-heavy operations: cosine similarity matrices, batch kNN, recall/MRR/NDCG metrics.

**Python layer** — orchestration: ChromaDB extraction, ground truth construction from knowledge graph, model wrappers, benchmark runner.

## Quick start

```bash
# Build Rust extension
maturin develop

# Run benchmark
python bench.py --models openai-small math-embed --k-values 5 10 20

# Run tests
cargo test          # Rust
python -m pytest    # Python
```

## Fine-tuning

```bash
# Train math-embed from scratch
python train.py --batch-size 8 --grad-accum 4 --max-seq-length 256 --epochs 3
```

See the [model card on HuggingFace](https://huggingface.co/RobBobin/math-embed) for full details.
