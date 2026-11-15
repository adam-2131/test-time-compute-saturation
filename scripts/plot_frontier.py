"""Plots for one fixed-k run, from the per-example CSVs it wrote.

    python scripts/plot_frontier.py
    python scripts/plot_frontier.py --experiments-dir experiments_gsm8k --plots-dir plots_gsm8k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # only save files, no window
import matplotlib.pyplot as plt  # noqa: E402

from analysis import discover_k_values

plt.rcParams.update({
    "figure.figsize": (9, 5.5),
    "savefig.dpi": 200,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
})


def log_xlim(k_values):
    # some room left and right so the outer labels are not cut off
    return min(k_values) * 0.55, max(k_values) * 1.9


def label_point(ax, x, y, text, *, dx=0, dy=12, ha="center", color="black"):
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        ha=ha,
        va="center",
        fontsize=10,
        color=color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
    )


def load_experiment_data(experiments_dir):
    return {
        k: pd.read_csv(experiments_dir / f"k_{k}.csv")
        for k in discover_k_values(experiments_dir)
    }


def summarize(data):
    rows = []
    for k in sorted(data):
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


def plot_accuracy_vs_k(summary, output_path):
    k = summary["k"].tolist()
    acc = summary["accuracy"].tolist()

    fig, ax = plt.subplots()
    ax.plot(k, acc, marker="o", markersize=9, linewidth=2.4, color="tab:blue", zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("k (number of samples, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Compute")
    ax.set_xticks(k)
    ax.set_xticklabels([str(v) for v in k])
    ax.set_xlim(*log_xlim(k))
    ax.set_ylim(0, min(1.02, max(acc) + 0.18))

    for i, (xv, yv) in enumerate(zip(k, acc)):
        ha = "left" if i == 0 else ("right" if i == len(k) - 1 else "center")
        dx = 6 if i == 0 else (-6 if i == len(k) - 1 else 0)
        label_point(ax, xv, yv, f"{yv:.3f}", dx=dx, dy=14, ha=ha, color="tab:blue")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


def plot_marginal_gain(summary, output_path):
    k = summary["k"].tolist()
    acc = summary["accuracy"].tolist()

    gains_pp = [(acc[i] - acc[i - 1]) * 100 for i in range(1, len(k))]
    labels = [f"{k[i-1]} \u2192 {k[i]}" for i in range(1, len(k))]

    fig, ax = plt.subplots()
    colors = ["tab:green" if g > 0 else "tab:red" for g in gains_pp]
    bars = ax.bar(labels, gains_pp, color=colors, edgecolor="#333333",
                  linewidth=0.6, width=0.6, zorder=3)

    ax.set_xlabel("k transition (10\u00d7 more compute)")
    ax.set_ylabel("Marginal accuracy gain (percentage points)")
    ax.set_title("Marginal Accuracy Gain per 10\u00d7 Compute")
    ax.axhline(y=0, color="#333333", linewidth=1.0)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    top = max(gains_pp + [0])
    bottom = min(gains_pp + [0])
    pad = max(0.6, (top - bottom) * 0.18)
    ax.set_ylim(bottom - pad, top + pad)

    for bar, g in zip(bars, gains_pp):
        height = bar.get_height()
        ax.annotate(
            f"{g:+.1f} pp",
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 8 if height >= 0 else -16),
            ha="center", fontsize=11, fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


def plot_effective_compute(summary, output_path):
    k = summary["k"].tolist()
    used = summary["avg_k_used"].tolist()

    fig, ax = plt.subplots()
    ax.plot(k, k, linestyle="--", linewidth=1.6, color="grey",
            label="No early stopping (k_used = k)", zorder=2)
    ax.plot(k, used, marker="s", markersize=9, linewidth=2.4, color="tab:purple",
            label="Actual samples used", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k_max (nominal, log scale)")
    ax.set_ylabel("Average samples used (log scale)")
    ax.set_title("Effective Compute: Early-Stopping Savings")
    ax.set_xticks(k)
    ax.set_xticklabels([str(v) for v in k])
    ax.set_xlim(*log_xlim(k))
    ax.set_ylim(min(used) * 0.6, max(k) * 2.2)
    ax.legend(loc="upper left")

    # labels go below the points, between the line and the diagonal
    for i, (xv, yv) in enumerate(zip(k, used)):
        if xv == 1:
            continue
        savings = (1 - yv / xv) * 100
        ha = "right" if i == len(k) - 1 else "center"
        dx = -8 if i == len(k) - 1 else 0
        label_point(ax, xv, yv, f"{savings:.0f}% saved", dx=dx, dy=-16, ha=ha, color="tab:purple")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


def main():
    parser = argparse.ArgumentParser(description="Plot the results of one fixed-k run")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--plots-dir", default="plots")
    args = parser.parse_args()

    try:
        data = load_experiment_data(Path(args.experiments_dir))
    except FileNotFoundError as e:
        sys.exit(f"{e}; run: python main.py --method fixed_k_majority")

    summary = summarize(data)

    print(f"{'k':>6}{'accuracy':>10}{'k_used':>9}{'conf':>7}{'n':>6}")
    for _, row in summary.iterrows():
        print(f"{int(row['k']):>6}{row['accuracy']:>10.4f}{row['avg_k_used']:>9.2f}"
              f"{row['avg_confidence']:>7.3f}{int(row['n_examples']):>6}")
    print()

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_vs_k(summary, plots_dir / "accuracy_vs_k.png")
    plot_marginal_gain(summary, plots_dir / "marginal_gain.png")
    plot_effective_compute(summary, plots_dir / "effective_compute.png")


if __name__ == "__main__":
    main()
