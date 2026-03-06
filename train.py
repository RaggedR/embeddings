#!/usr/bin/env python3
"""Fine-tune an embedding model for mathematical paper retrieval.

Uses contrastive learning (MultipleNegativesRankingLoss) with Matryoshka
dimensionality reduction, trained on (concept, chunk) pairs extracted
from a knowledge graph + ChromaDB.

Usage:
    python train.py                          # defaults: specter2 base, 3 epochs
    python train.py --base-model malteos/scincl --epochs 5
    python train.py --matryoshka-dims 768 512 256 128
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

from embench.training_data import build_dataset


DEFAULT_BASE_MODEL = "allenai/specter2_base"
DEFAULT_OUTPUT_DIR = "models/math-embed"


def build_ir_evaluator(
    kg_path: str,
    chroma_path: str,
    min_degree: int = 2,
) -> InformationRetrievalEvaluator:
    """Build an IR evaluator from the ground truth for training-time monitoring."""
    from embench.extract import extract_collection
    from embench.ground_truth import build_ground_truth

    data = extract_collection(chroma_path)
    ground_truth = build_ground_truth(
        kg_path, data["source_to_indices"], min_degree=min_degree
    )

    # sentence-transformers IR evaluator format:
    #   queries: {qid: query_text}
    #   corpus: {cid: corpus_text}
    #   relevant_docs: {qid: {cid: 1}}
    queries = {}
    relevant_docs = {}
    corpus = {str(i): doc for i, doc in enumerate(data["documents"])}

    for i, gt in enumerate(ground_truth):
        qid = f"q{i}"
        queries[qid] = gt["query"]
        relevant_docs[qid] = {str(idx): 1 for idx in gt["relevant_indices"]}

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="math-retrieval",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[5, 10, 20],
        precision_recall_at_k=[20],
        show_progress_bar=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune embedding model for math paper retrieval"
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"HuggingFace model ID to fine-tune (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for trained model (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--chroma-path",
        default="/Users/robin/data/arxiv-rag/chroma_db",
    )
    parser.add_argument(
        "--kg-path",
        default="/Users/robin/data/arxiv-rag/knowledge_graph.json",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--matryoshka-dims",
        nargs="+",
        type=int,
        default=[768, 512, 256, 128],
        help="Matryoshka truncation dimensions (default: 768 512 256 128)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Max token sequence length (default: 256, BERT max is 512)",
    )
    parser.add_argument(
        "--no-edges",
        action="store_true",
        help="Exclude cross-concept edge pairs from training data",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="Evaluate every N steps (default: 200)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training (slower but no memory limits)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )
    args = parser.parse_args()

    # Build training data
    print("Building training dataset...")
    ds = build_dataset(
        kg_path=args.kg_path,
        chroma_path=args.chroma_path,
        include_edges=not args.no_edges,
    )
    print(f"  Train: {ds['stats']['train_pairs']} pairs")
    print(f"  Val: {ds['stats']['val_pairs']} pairs")
    print(f"  Unique anchors: {ds['stats']['unique_anchors']}")

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(ds["train"])
    val_dataset = Dataset.from_list(ds["val"])

    # Force CPU if requested
    if args.cpu:
        import torch
        device = "cpu"
    else:
        device = None  # auto-detect

    # Load model
    print(f"\nLoading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_seq_length
    print(f"  Dimension: {model.get_sentence_embedding_dimension()}")
    print(f"  Max seq length: {model.max_seq_length}")
    print(f"  Device: {model.device}")

    # Loss: MNRL wrapped in Matryoshka
    base_loss = MultipleNegativesRankingLoss(model)
    loss = MatryoshkaLoss(model, base_loss, matryoshka_dims=args.matryoshka_dims)
    print(f"  Matryoshka dims: {args.matryoshka_dims}")

    # Training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=False,
        bf16=False,
        use_cpu=args.cpu,
        dataloader_pin_memory=False,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        batch_sampler="no_duplicates",
        logging_steps=10,
    )

    # Train
    print(f"\nStarting training ({args.epochs} epochs, batch_size={args.batch_size})...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
    )

    trainer.train()

    # Save final model
    final_path = Path(args.output_dir) / "final"
    model.save(str(final_path))
    print(f"\nModel saved to {final_path}")

    # Save training config
    config = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "matryoshka_dims": args.matryoshka_dims,
        "dataset_stats": ds["stats"],
    }
    config_path = Path(args.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
