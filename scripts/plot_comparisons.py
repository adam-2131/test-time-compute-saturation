"""Comparison plots across the three runs, saved in plots_combined/:
the two models on AIME, and AIME vs. GSM8K for gpt-4o-mini.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # only save files, no window
import matplotlib.pyplot as plt  # noqa: E402

from analysis import accuracy, discover_k_values, load_k

RUNS = {
    "aime_4o": Path("experiments"),
    "gsm8k_4o": Path("experiments_gsm8k"),
    "aime_41": Path("experiments_aime_41mini"),
}

OUT_DIR = "plots_combined"

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


def load_accuracy(run_dir, k_values):
    accs = []
    for k in k_values:
        rows = load_k(run_dir, k)
        accs.append(accuracy(rows, list(rows)))
    return accs


def _style_ax(ax, title, k_values):
    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlim(min(k_values) * 0.55, max(k_values) * 1.9)
    ax.set_xlabel("k (number of samples, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.legend(loc="center right")


def annotate(ax, xs, ys, color, dy):
    # dy moves the labels up or down so the two lines don't overlap
    for i, (x, y) in enumerate(zip(xs, ys)):
        ha = "left" if i == 0 else ("right" if i == len(xs) - 1 else "center")
        dx = 6 if i == 0 else (-6 if i == len(xs) - 1 else 0)
        ax.annotate(
            f"{y:.3f}", (x, y),
            textcoords="offset points", xytext=(dx, dy),
            ha=ha, va="center", fontsize=9.5, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
        )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # only the k values that all runs have
    k_values = sorted(set.intersection(*(set(discover_k_values(d)) for d in RUNS.values())))

    aime_4o = load_accuracy(RUNS["aime_4o"], k_values)
    gsm8k_4o = load_accuracy(RUNS["gsm8k_4o"], k_values)
    aime_41 = load_accuracy(RUNS["aime_41"], k_values)

    # the two models are close together, so one gets labels above, one below
    fig, ax = plt.subplots()
    ax.plot(k_values, aime_41, marker="s", markersize=9, linewidth=2.4,
            color="tab:orange", label="gpt-4.1-mini", zorder=3)
    ax.plot(k_values, aime_4o, marker="o", markersize=9, linewidth=2.4,
            color="tab:blue", label="gpt-4o-mini", zorder=3)
    annotate(ax, k_values, aime_41, "tab:orange", dy=14)
    annotate(ax, k_values, aime_4o, "tab:blue", dy=-15)
    _style_ax(ax, "AIME: accuracy vs k, by model", k_values)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "model_comparison_aime.png")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    print(p1)

    fig, ax = plt.subplots()
    ax.plot(k_values, gsm8k_4o, marker="o", markersize=9, linewidth=2.4,
            color="tab:green", label="GSM8K (easier)", zorder=3)
    ax.plot(k_values, aime_4o, marker="s", markersize=9, linewidth=2.4,
            color="tab:red", label="AIME (harder)", zorder=3)
    annotate(ax, k_values, gsm8k_4o, "tab:green", dy=14)
    annotate(ax, k_values, aime_4o, "tab:red", dy=14)
    _style_ax(ax, "gpt-4o-mini: accuracy vs k, by benchmark difficulty", k_values)
    ax.legend(loc="center left")
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "difficulty_comparison.png")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    print(p2)


if __name__ == "__main__":
    main()
