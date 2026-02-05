#!/usr/bin/env python3
"""
Post-hoc plotting for fixed-k majority voting experiments.

Reads per-example CSVs from experiments/k_*.csv and generates:
  - plots/accuracy_vs_k.png
  - plots/marginal_gain.png
  - plots/effective_compute.png

Usage:
    python scripts/plot_frontier.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is on path for imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

# Expected k values and paths
EXPECTED_K_VALUES = [1, 10, 100, 1000]
EXPERIMENTS_DIR = _REPO_ROOT / "experiments"
PLOTS_DIR = _REPO_ROOT / "plots"


def load_experiment_data(k_values: list[int] = EXPECTED_K_VALUES) -> dict[int, pd.DataFrame]:
    """
    Load per-example CSVs for each k value.
    
    Raises FileNotFoundError with clear message if any expected CSV is missing.
    """
    data: dict[int, pd.DataFrame] = {}
    missing: list[str] = []
    
    for k in k_values:
        csv_path = EXPERIMENTS_DIR / f"k_{k}.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))
        else:
            df = pd.read_csv(csv_path)
            data[k] = df
    
    if missing:
        raise FileNotFoundError(
            f"Missing experiment CSVs. Run the fixed-k experiments first:\n"
            f"  python main.py --method fixed_k_majority\n\n"
            f"Missing files:\n  " + "\n  ".join(missing)
        )
    
    return data


def compute_summary_stats(data: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute summary statistics for each k value.
    
    Returns DataFrame with columns: k, accuracy, avg_k_used, avg_confidence, avg_entropy, n_examples
    """
    rows = []
    for k in sorted(data.keys()):
        df = data[k]
        rows.append({
            "k": k,
            "accuracy": df["correct"].mean(),
            "avg_k_used": df["k_used"].mean(),
            "avg_confidence": df["confidence"].mean(),
            "avg_entropy": df["entropy"].mean(),
            "n_examples": len(df),
        })
    return pd.DataFrame(rows)


def plot_accuracy_vs_k(summary: pd.DataFrame, output_path: Path) -> None:
    """
    Generate accuracy vs k plot (log scale x-axis).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(
        summary["k"],
        summary["accuracy"],
        marker="o",
        markersize=8,
        linewidth=2,
        color="#2563eb",
        label="Accuracy",
    )
    
    ax.set_xscale("log")
    ax.set_xlabel("k (number of samples)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Compute (Fixed-k Majority Voting)", fontsize=14)
    
    # Set x-ticks to actual k values
    ax.set_xticks(summary["k"].tolist())
    ax.set_xticklabels([str(k) for k in summary["k"]])
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.05)
    
    # Annotate points with accuracy values
    for _, row in summary.iterrows():
        ax.annotate(
            f"{row['accuracy']:.3f}",
            (row["k"], row["accuracy"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


def plot_marginal_gain(summary: pd.DataFrame, output_path: Path) -> None:
    """
    Generate bar plot of marginal accuracy gains between consecutive k values.
    """
    # Compute marginal gains
    k_values = summary["k"].tolist()
    accuracies = summary["accuracy"].tolist()
    
    gains = []
    labels = []
    for i in range(1, len(k_values)):
        gain = accuracies[i] - accuracies[i - 1]
        gains.append(gain)
        labels.append(f"{k_values[i]}−{k_values[i-1]}")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ["#22c55e" if g > 0 else "#ef4444" for g in gains]
    bars = ax.bar(labels, gains, color=colors, edgecolor="black", linewidth=0.5)
    
    ax.set_xlabel("k transition", fontsize=12)
    ax.set_ylabel("Marginal accuracy gain", fontsize=12)
    ax.set_title("Marginal Accuracy Gain per 10× Compute", fontsize=14)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color="black", linewidth=0.8)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    
    # Annotate bars with values
    for bar, gain in zip(bars, gains):
        height = bar.get_height()
        ax.annotate(
            f"{gain:+.3f}",
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 5 if height >= 0 else -15),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


def plot_effective_compute(summary: pd.DataFrame, output_path: Path) -> None:
    """
    Generate plot of effective compute (avg_k_used) vs nominal k.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot avg_k_used
    ax.plot(
        summary["k"],
        summary["avg_k_used"],
        marker="s",
        markersize=8,
        linewidth=2,
        color="#8b5cf6",
        label="Avg k_used (actual)",
    )
    
    # Plot diagonal (k_used = k) for reference
    ax.plot(
        summary["k"],
        summary["k"],
        linestyle="--",
        linewidth=1.5,
        color="#9ca3af",
        label="k_used = k (no early stop)",
    )
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k_max (nominal)", fontsize=12)
    ax.set_ylabel("avg_k_used (actual)", fontsize=12)
    ax.set_title("Effective Compute: Early Stopping Savings", fontsize=14)
    
    # Set x-ticks to actual k values
    ax.set_xticks(summary["k"].tolist())
    ax.set_xticklabels([str(k) for k in summary["k"]])
    
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    # Annotate with savings percentage
    for _, row in summary.iterrows():
        savings = (1 - row["avg_k_used"] / row["k"]) * 100
        if row["k"] > 1:  # Skip k=1 annotation
            ax.annotate(
                f"{savings:.1f}% saved",
                (row["k"], row["avg_k_used"]),
                textcoords="offset points",
                xytext=(10, -5),
                ha="left",
                fontsize=9,
                color="#8b5cf6",
            )
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


def print_summary_table(summary: pd.DataFrame) -> None:
    """
    Print summary statistics table to stdout.
    """
    print()
    print("=" * 70)
    print("Fixed-k Majority Voting Summary Statistics")
    print("=" * 70)
    
    header = f"{'k':>8}  {'accuracy':>10}  {'avg_k_used':>12}  {'avg_conf':>10}  {'n':>6}"
    print(header)
    print("-" * len(header))
    
    for _, row in summary.iterrows():
        print(
            f"{int(row['k']):>8}  "
            f"{row['accuracy']:>10.4f}  "
            f"{row['avg_k_used']:>12.2f}  "
            f"{row['avg_confidence']:>10.4f}  "
            f"{int(row['n_examples']):>6}"
        )
    
    print()


def main() -> None:
    print("Loading experiment data...")
    
    # Load data
    try:
        data = load_experiment_data()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Loaded {len(data)} experiment files")
    
    # Compute summary statistics
    summary = compute_summary_stats(data)
    
    # Print summary table
    print_summary_table(summary)
    
    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Generating plots...")
    
    # Generate plots
    plot_accuracy_vs_k(summary, PLOTS_DIR / "accuracy_vs_k.png")
    plot_marginal_gain(summary, PLOTS_DIR / "marginal_gain.png")
    plot_effective_compute(summary, PLOTS_DIR / "effective_compute.png")
    
    print()
    print(f"All plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
