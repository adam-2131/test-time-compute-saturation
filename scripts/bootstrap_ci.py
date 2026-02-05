#!/usr/bin/env python3
"""
LEGACY: Bootstrap 95% confidence intervals for token-budgeted experiments.

This script operates on artifacts from the legacy token-budgeted experiments:
  results/per_problem_results.jsonl

The primary experiment path is now fixed-k majority voting (--method fixed_k_majority),
which outputs per-example CSVs to experiments/k_*.csv.

Keep for backwards compatibility with existing token-budgeted results.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any


IN_PATH = os.path.join("results", "per_problem_results.jsonl")


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile on a sorted list, p in [0, 100]."""
    if not sorted_vals:
        raise ValueError("empty values")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[k]


def bootstrap_mean_ci(
    xs: list[int],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Bootstrap CI for mean of 0/1 values.
    Returns: (point_estimate, lo, hi)
    """
    if not xs:
        raise ValueError("empty sample list")
    n = len(xs)
    point = sum(xs) / n

    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += xs[rng.randrange(n)]
        stats.append(s / n)
    stats.sort()

    lo = _percentile(stats, 100.0 * (alpha / 2.0))
    hi = _percentile(stats, 100.0 * (1.0 - alpha / 2.0))
    return point, lo, hi


def bootstrap_paired_diff_ci(
    xs_a: list[int],
    xs_b: list[int],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Paired bootstrap CI for difference of means: mean(B - A) over aligned problems.
    Returns: (point_diff, lo, hi)
    """
    if len(xs_a) != len(xs_b):
        raise ValueError(f"length mismatch: {len(xs_a)} vs {len(xs_b)}")
    if not xs_a:
        raise ValueError("empty sample list")
    n = len(xs_a)
    point = (sum(xs_b) - sum(xs_a)) / n

    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            j = rng.randrange(n)
            s += (xs_b[j] - xs_a[j])
        stats.append(s / n)
    stats.sort()
    lo = _percentile(stats, 100.0 * (alpha / 2.0))
    hi = _percentile(stats, 100.0 * (1.0 - alpha / 2.0))
    return point, lo, hi


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found (run main.py once to create it)")
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None, help="Run id to analyze (default: latest in file).")
    ap.add_argument("--metric", default="accuracy", choices=["accuracy", "pass_at_k", "majority_correct"])
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--diff", default="10,100", help="Compute paired diff CI between multipliers A,B (default: 10,100)")
    args = ap.parse_args()

    rows = _load_jsonl(IN_PATH)
    if not rows:
        raise RuntimeError(f"{IN_PATH} has no rows")

    run_ids = [r.get("run_id") for r in rows if r.get("run_id")]
    if not run_ids:
        raise RuntimeError("no run_id entries found")
    target_run_id = args.run_id or run_ids[-1]

    filtered = [r for r in rows if r.get("run_id") == target_run_id]
    if not filtered:
        raise RuntimeError(f"run_id={target_run_id} not found in {IN_PATH}")

    # Group by (dataset, method, seed) within the run_id
    groups: dict[tuple[str, str, int], dict[int, list[int]]] = defaultdict(dict)
    meta: dict[tuple[str, str, int], dict[str, Any]] = {}

    metric_key = {
        "accuracy": "per_problem_correct",
        "pass_at_k": "per_problem_pass_at_k",
        "majority_correct": "per_problem_majority_correct",
    }[args.metric]

    for r in filtered:
        dataset = str(r.get("dataset", ""))
        method = str(r.get("method", ""))
        seed = int(r.get("seed", 0))
        m = int(r.get("compute_multiplier", 0))
        xs = r.get(metric_key)
        if not isinstance(xs, list):
            continue
        xs_int = [1 if bool(v) else 0 for v in xs]
        groups[(dataset, method, seed)][m] = xs_int
        meta[(dataset, method, seed)] = {
            "model": r.get("model", ""),
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "num_problems": len(xs_int),
        }

    diff_a, diff_b = (int(x.strip()) for x in str(args.diff).split(","))

    out: dict[str, Any] = {
        "run_id": target_run_id,
        "metric": args.metric,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "alpha": args.alpha,
        "groups": [],
    }

    for key, by_m in sorted(groups.items()):
        if not by_m:
            continue
        ds, method, seed = key
        group_out: dict[str, Any] = {
            **meta.get(key, {}),
            "multipliers": {},
            "diffs": {},
        }

        for m, xs in sorted(by_m.items()):
            point, lo, hi = bootstrap_mean_ci(xs, n_boot=args.n_bootstrap, seed=args.seed, alpha=args.alpha)
            group_out["multipliers"][str(m)] = {"point": point, "ci95": [lo, hi]}

        if diff_a in by_m and diff_b in by_m:
            point, lo, hi = bootstrap_paired_diff_ci(
                by_m[diff_a],
                by_m[diff_b],
                n_boot=args.n_bootstrap,
                seed=args.seed,
                alpha=args.alpha,
            )
            group_out["diffs"][f"{diff_b}-{diff_a}"] = {"point": point, "ci95": [lo, hi]}

        out["groups"].append(group_out)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"bootstrap_ci_{target_run_id}_{args.metric}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()













