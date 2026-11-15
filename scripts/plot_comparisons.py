
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis import PLOT_STYLE, accuracy, discover_k_values, load_k, log_xlim


RUNS = {
    "aime_4o": Path("experiments"),
    "gsm8k_4o": Path("experiments_gsm8k"),
    "aime_41": Path("experiments_aime_41mini"),
}

OUT_DIR = Path("plots_combined")

plt.rcParams.update(PLOT_STYLE)


def load_accuracy(run_dir, k_values):
    accs = []

    for k in k_values:
        rows = load_k(run_dir, k)
        accs.append(accuracy(rows, list(rows)))

    return accs


def style_ax(ax, title, k_values):
    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlim(*log_xlim(k_values))
    ax.set_xlabel("k (number of samples, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)


def annotate(ax, xs, ys, color, dy):
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i == 0:
            ha, dx = "left", 6
        elif i == len(xs) - 1:
            ha, dx = "right", -6
        else:
            ha, dx = "center", 0

        ax.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va="center",
            fontsize=9.5,
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.75,
            ),
        )


def save(fig, name):
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    k_values = sorted(set.intersection(
        *(set(discover_k_values(d)) for d in RUNS.values())
    ))

    aime_4o = load_accuracy(RUNS["aime_4o"], k_values)
    gsm8k_4o = load_accuracy(RUNS["gsm8k_4o"], k_values)
    aime_41 = load_accuracy(RUNS["aime_41"], k_values)

    fig, ax = plt.subplots()

    ax.plot(
        k_values, aime_41,
        marker="s", markersize=9, linewidth=2.4,
        color="tab:orange", label="gpt-4.1-mini", zorder=3,
    )
    ax.plot(
        k_values, aime_4o,
        marker="o", markersize=9, linewidth=2.4,
        color="tab:blue", label="gpt-4o-mini", zorder=3,
    )

    annotate(ax, k_values, aime_41, "tab:orange", dy=14)
    annotate(ax, k_values, aime_4o, "tab:blue", dy=-15)

    style_ax(ax, "AIME: accuracy vs k, by model", k_values)
    ax.legend(loc="center right")
    save(fig, "model_comparison_aime.png")

    fig, ax = plt.subplots()

    ax.plot(
        k_values, gsm8k_4o,
        marker="o", markersize=9, linewidth=2.4,
        color="tab:green", label="GSM8K (easier)", zorder=3,
    )
    ax.plot(
        k_values, aime_4o,
        marker="s", markersize=9, linewidth=2.4,
        color="tab:red", label="AIME (harder)", zorder=3,
    )

    annotate(ax, k_values, gsm8k_4o, "tab:green", dy=14)
    annotate(ax, k_values, aime_4o, "tab:red", dy=14)

    style_ax(ax, "gpt-4o-mini: accuracy vs k, by benchmark difficulty", k_values)
    ax.legend(loc="center left")
    save(fig, "difficulty_comparison.png")


if __name__ == "__main__":
    main()
