#!/usr/bin/env python3
"""CLI entrypoint for embedding benchmark."""

import argparse
import sys

import numpy as np


AVAILABLE_MODELS = {
    "openai-small": "OpenAI text-embedding-3-small (1536-dim, baseline)",
    "openai-large": "OpenAI text-embedding-3-large (3072-dim)",
    "bge-large": "BAAI/bge-large-en-v1.5 via MLX (1024-dim)",
    "qwen3-0.6b": "Qwen3-Embedding-0.6B via MLX (1024-dim)",
}


def make_model(name: str):
    """Instantiate a model by name."""
    if name == "openai-small":
        from embench.models.openai_embed import OpenAISmall
        return OpenAISmall()
    elif name == "openai-large":
        from embench.models.openai_embed import OpenAILarge
        return OpenAILarge()
    elif name == "bge-large":
        from embench.models.mlx_embed import BGELarge
        return BGELarge()
    elif name == "qwen3-0.6b":
        from embench.models.mlx_embed import Qwen3Embed
        return Qwen3Embed()
    else:
        print(f"Unknown model: {name}")
        print(f"Available: {', '.join(AVAILABLE_MODELS)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark embedding models for math paper retrieval",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["openai-small"],
        choices=list(AVAILABLE_MODELS),
        help="Models to benchmark (default: openai-small)",
    )
    parser.add_argument(
        "--chroma-path",
        default="/Users/robin/data/arxiv-rag/chroma_db",
        help="Path to ChromaDB directory",
    )
    parser.add_argument(
        "--kg-path",
        default="/Users/robin/data/arxiv-rag/knowledge_graph.json",
        help="Path to knowledge_graph.json",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[5, 10, 20],
        help="K values for Recall@K (default: 5 10 20)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--min-degree",
        type=int,
        default=2,
        help="Minimum papers per concept for ground truth (default: 2)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots",
    )
    args = parser.parse_args()

    from embench.bench import BenchmarkRunner
    from embench.report import print_comparison_table, save_json, plot_recall_curves

    runner = BenchmarkRunner(
        chroma_path=args.chroma_path,
        kg_path=args.kg_path,
        output_dir=args.output_dir,
        min_degree=args.min_degree,
        k_values=args.k_values,
    )

    # Build model list with optional pre-computed embeddings
    models = []
    for name in args.models:
        model = make_model(name)

        # For openai-small, use pre-computed embeddings from ChromaDB
        precomputed = None
        if name == "openai-small":
            runner._ensure_data()
            precomputed = runner._data["embeddings"]
            print(f"Using ChromaDB embeddings as baseline for {name}")

        models.append((model, precomputed))

    results = runner.run_all(models)

    print_comparison_table(results)
    save_json(results, args.output_dir)

    if not args.no_plot:
        plot_recall_curves(results, args.output_dir)


if __name__ == "__main__":
    main()
