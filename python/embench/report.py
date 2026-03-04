"""Reporting: comparison tables, JSON export, plots."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from embench.bench import ModelResult


def print_comparison_table(results: list[ModelResult]):
    """Print a formatted comparison table to stdout."""
    if not results:
        print("No results to display.")
        return

    k_values = results[0].k_values

    # Header
    cols = ["Model", "Dim"]
    for k in k_values:
        cols.append(f"R@{k}")
    cols.extend(["MRR", "NDCG@10", "Embed(s)"])

    widths = [max(len(c), 12) for c in cols]
    widths[0] = max(widths[0], max(len(r.model_name) for r in results))

    header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "-+-".join("-" * w for w in widths)

    print(f"\n{header}")
    print(sep)

    for r in results:
        row = [r.model_name, str(r.dim)]
        for k in k_values:
            val = r.metrics.get(f"recall@{k}", 0.0)
            row.append(f"{val:.4f}")
        row.append(f"{r.metrics.get('mrr', 0.0):.4f}")
        row.append(f"{r.metrics.get('ndcg@10', 0.0):.4f}")
        row.append(f"{r.embed_time_s:.1f}" if r.embed_time_s > 0 else "pre")
        print(" | ".join(v.ljust(w) for v, w in zip(row, widths)))


def save_json(results: list[ModelResult], output_dir: str) -> Path:
    """Save results as JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"benchmark_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "models": [
            {
                "name": r.model_name,
                "dim": r.dim,
                "metrics": r.metrics,
                "embed_time_s": r.embed_time_s,
                "eval_time_s": r.eval_time_s,
                "k_values": r.k_values,
            }
            for r in results
        ],
    }

    path.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to {path}")
    return path


def plot_recall_curves(results: list[ModelResult], output_dir: str) -> Path | None:
    """Plot Recall@K curves for all models."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for r in results:
        ks = sorted(r.k_values)
        recalls = [r.metrics.get(f"recall@{k}", 0.0) for k in ks]
        ax.plot(ks, recalls, marker="o", label=r.model_name)

    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title("Embedding Model Comparison — Math Paper Retrieval")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"recall_curves_{timestamp}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")
    return path
