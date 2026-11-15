"""Helpers shared by the analysis and plot scripts.

A run is a folder with one CSV per k value (k_1.csv, k_10.csv, ...), as
written by main.py.
"""

from __future__ import annotations

import csv
from math import comb

# matplotlib settings used by both plot scripts
PLOT_STYLE = {
    "figure.figsize": (9, 5.5),
    "savefig.dpi": 200,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
}


def log_xlim(k_values):
    # some room left and right so the outer labels are not cut off
    return min(k_values) * 0.55, max(k_values) * 1.9


def discover_k_values(run_dir):
    """Sorted k values found in the file names of a run folder."""
    ks = []
    for path in run_dir.glob("k_*.csv"):
        try:
            ks.append(int(path.stem.removeprefix("k_")))
        except ValueError:
            continue
    if not ks:
        raise FileNotFoundError(f"no k_*.csv files in {run_dir}")
    return sorted(ks)


def is_correct(row):
    return row["correct"] == "1"


def load_k(run_dir, k):
    """Rows of one k value, by example_id."""
    with (run_dir / f"k_{k}.csv").open(encoding="utf-8") as fh:
        return {row["example_id"]: row for row in csv.DictReader(fh)}


def load_run(run_dir):
    """All k values of a run, plus the ids that appear in every one of them
    (so an unfinished run doesn't compare different problems)."""
    by_k = {k: load_k(run_dir, k) for k in discover_k_values(run_dir)}
    shared = sorted(set.intersection(*(set(rows) for rows in by_k.values())))
    return by_k, shared


def shared_ids(*runs):
    """Ids that appear in all runs, in the order of the first one."""
    if not runs:
        return []
    others = [set(rows) for rows in runs[1:]]
    return [i for i in runs[0] if all(i in other for other in others)]


def accuracy(rows, ids):
    return sum(is_correct(rows[i]) for i in ids) / len(ids)


def mcnemar_exact(n_lost, n_won):
    """Two-sided exact McNemar test. Only the problems that changed matter,
    so it is a binomial test with p = 0.5 on those."""
    discordant = n_lost + n_won
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, i) for i in range(min(n_lost, n_won) + 1)) / 2 ** discordant
    return min(1.0, 2 * tail)


def flips(before, after, ids):
    """Ids that went correct -> wrong, and wrong -> correct."""
    lost = [i for i in ids if is_correct(before[i]) and not is_correct(after[i])]
    won = [i for i in ids if not is_correct(before[i]) and is_correct(after[i])]
    return lost, won
