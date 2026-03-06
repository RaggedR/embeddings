# Feature: Embedding Benchmark & Fine-Tuning Pipeline
> Benchmarks embedding models for mathematical paper retrieval and fine-tunes a domain-specific model using knowledge-graph-guided contrastive learning.

## Overview
INSTINCT uses `text-embedding-3-small` (OpenAI, 1536-dim) for all retrieval. This project:
1. **Benchmarks** off-the-shelf embedding models against a knowledge-graph-derived ground truth
2. **Fine-tunes** a domain-specific embedding model that dramatically outperforms all baselines

The fine-tuned model (`math-embed`) achieves **MRR 0.816** vs OpenAI's 0.461 — a 77% improvement.

## Results

| Model | MRR | NDCG@10 | Recall@5 | Recall@10 | Recall@20 |
|-------|-----|---------|----------|-----------|-----------|
| **math-embed** | **0.816** | **0.736** | **0.064** | **0.068** | **0.072** |
| openai-small | 0.461 | 0.324 | 0.024 | 0.031 | 0.040 |
| specter2 | 0.360 | 0.225 | 0.015 | 0.021 | 0.029 |
| scincl | 0.306 | 0.205 | 0.013 | 0.019 | 0.027 |

Recall appears low because relevant sets are large (often 100+ chunks per concept). **MRR and NDCG are the meaningful metrics.**

## Architecture

### Benchmarking layer
- **Rust** (`rust/emb-metrics/src/`): cosine similarity, batch kNN, Recall/MRR/NDCG (rayon-parallelized)
- **Python** (`python/embench/`): model wrappers, ChromaDB extraction, ground truth generation, reporting
- **CLI**: `bench.py` — run benchmarks across models

### Training layer
- **Training data** (`python/embench/training_data.py`): Generates (anchor, positive) pairs from KG
- **Fine-tuning** (`train.py`): Contrastive learning with MNRL + MatryoshkaLoss
- **Trained model**: `models/math-embed/final/` (also on HuggingFace)

## How the Training Pipeline Works

### Step 1: Training data from knowledge graph
`python/embench/training_data.py` generates contrastive pairs:

- **Direct pairs**: concept name/description → chunks from that concept's papers
- **Edge pairs**: cross-concept pairs from KG edges (e.g., if concept A "generalizes" B, pair A's name with B's chunks)
- **Deduplication** and 90/10 train/val split

With the math papers KG: 25,121 total pairs (21,544 direct + 4,855 edge), 1,114 unique anchors.

### Step 2: Fine-tuning with contrastive loss
`train.py` fine-tunes SPECTER2 base (a SciBERT variant pre-trained on 6M citation triplets):

```bash
python train.py \
  --base-model allenai/specter2_base \
  --batch-size 8 \
  --max-seq-length 256 \
  --grad-accum 4 \
  --epochs 3 \
  --matryoshka-dims 768 512 256 128
```

- **Loss**: MultipleNegativesRankingLoss wrapped in MatryoshkaLoss
- **MNRL**: In-batch negatives (other items in the batch serve as negatives automatically)
- **Matryoshka**: Trains the model so that prefix subsets of the embedding vector are themselves useful embeddings (768, 512, 256, 128 dims)
- **Hardware**: ~4 hours on Apple M-series GPU (MPS). Batch size 8 + grad accumulation 4 = effective batch size 32.

### Step 3: Benchmark the result
```bash
python bench.py --models openai-small specter2 scincl math-embed --no-plot
```

## Repeating for a New Paper Collection

**This is the key section for future Claude sessions.** To repeat this process with different papers (e.g., puzzle papers):

### Prerequisites
1. A ChromaDB collection of paper chunks with OpenAI embeddings (from INSTINCT's ingestion pipeline)
2. A knowledge graph JSON file for those papers (built by `/knowledge-graph` or `build_knowledge_graph.py`)
3. Python environment with: `sentence-transformers`, `torch`, `datasets`, `chromadb`

### Steps
1. **Point to your data**:
   ```bash
   export CHROMA_PATH="/path/to/your/chroma_db"
   export KG_PATH="/path/to/your/knowledge_graph.json"
   ```

2. **Check training data size**:
   ```python
   from embench.training_data import build_dataset
   ds = build_dataset(kg_path=KG_PATH, chroma_path=CHROMA_PATH)
   print(ds["stats"])  # Should have 1000+ pairs for good results
   ```

3. **Fine-tune** (adjust batch-size for your hardware):
   ```bash
   python train.py \
     --chroma-path $CHROMA_PATH \
     --kg-path $KG_PATH \
     --batch-size 8 --grad-accum 4 --max-seq-length 256 \
     --epochs 3 \
     --output-dir models/puzzle-embed
   ```

4. **Register the model** in `bench.py`:
   Add an entry to `AVAILABLE_MODELS` and `make_model()` pointing to `models/puzzle-embed/final`.

5. **Benchmark**:
   ```bash
   python bench.py --models openai-small puzzle-embed \
     --chroma-path $CHROMA_PATH --kg-path $KG_PATH --no-plot
   ```

### Memory notes (Apple Silicon)
- MPS (GPU) with batch_size=8, max_seq_length=256 works well (~5s/step)
- batch_size=16+ or max_seq_length=512 will likely OOM on 16GB machines
- `--cpu` flag works but is ~25x slower
- gradient_accumulation_steps increases effective batch size without more memory

### Model lineage
```
BERT (Google, 110M params, general English)
  └─ SciBERT (Allen AI, retrained on scientific papers)
      └─ SPECTER2 base (Allen AI, + 6M citation triplets)
          └─ math-embed (us, + KG-derived concept-chunk pairs)
```

## Resources
- Data: `~/data/arxiv-rag/` (4,794 chunks in ChromaDB, 559 concepts in KG)
- Parent project: INSTINCT (`~/git/math-research-tools/`)
- Research paper: `paper/math_embeddings.tex` (compiled to `paper/math_embeddings.pdf`)
- HuggingFace model: `RobBobin/math-embed` (768-dim, sentence-transformers compatible)

## Assets
- Rust crate: `rust/emb-metrics/src/`
- Python package: `python/embench/`
- CLI entrypoint: `bench.py`
- Training script: `train.py`
- Training data generator: `python/embench/training_data.py`
- Model wrappers: `python/embench/models/scientific_embed.py`
- Trained model: `models/math-embed/final/`
- Tests: `tests/`
