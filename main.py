
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from statistics import fmean

from data import load_jsonl_dataset
from eval import run_fixed_k_majority


DEFAULT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7
MAX_TOKENS = 2048

COLUMNS = [
    "example_id",
    "k_max",
    "k_used",
    "prediction",
    "confidence",
    "entropy",
    "correct",
    "runtime_seconds",
]


def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def select_subset(problems, subset_size, subset_seed):
    if len(problems) <= subset_size:
        return problems

    rng = random.Random(subset_seed)
    indices = list(range(len(problems)))
    rng.shuffle(indices)

    return [problems[i] for i in sorted(indices[:subset_size])]


def write_csv(path, results):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()

        for r in results:
            writer.writerow({
                "example_id": r["example_id"],
                "k_max": r["k_max"],
                "k_used": r["k_used"],
                "prediction": "" if r["prediction"] is None else r["prediction"],
                "confidence": f"{r['confidence']:.6f}",
                "entropy": f"{r['entropy']:.6f}",
                "correct": int(r["correct"]),
                "runtime_seconds": f"{r['runtime_seconds']:.4f}",
            })


def run_fixed_k(args):
    problems = select_subset(
        load_jsonl_dataset(args.dataset_path),
        args.subset_size,
        args.subset_seed,
    )

    if not problems:
        raise ValueError(f"no problems loaded from {args.dataset_path}")

    k_values = parse_int_list(args.k_values)
    out_dir = Path(args.per_example_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"fixed-k majority voting on {Path(args.dataset_path).stem}, model {args.model}")
    print(f"{len(problems)} problems (subset_seed={args.subset_seed}), k = {k_values}")
    print(
        f"early stop: confidence {args.early_stop_confidence}, "
        f"stability {args.early_stop_stability}, "
        f"min samples {args.early_stop_min_samples}\n"
    )

    summary = []

    for k in k_values:
        print(f"k_max={k}")

        results = run_fixed_k_majority(
            problems,
            args.model,
            TEMPERATURE,
            MAX_TOKENS,
            k_max=k,
            seed=args.seed,
            early_stop_stability=args.early_stop_stability,
            early_stop_confidence=args.early_stop_confidence,
            early_stop_min_samples=args.early_stop_min_samples,
            max_workers=args.max_workers,
            progress=True,
        )

        path = out_dir / f"k_{k}.csv"
        write_csv(path, results)
        print(f"wrote {path} ({len(results)} rows)")

        summary.append((
            k,
            fmean(r["correct"] for r in results),
            fmean(r["k_used"] for r in results),
            fmean(r["confidence"] for r in results),
            fmean(r["runtime_seconds"] for r in results),
        ))

    print(f"\n{'k_max':>6}{'accuracy':>10}{'k_used':>9}{'conf':>7}{'seconds':>9}")

    for k, accuracy, k_used, confidence, runtime in summary:
        print(f"{k:>6}{accuracy:>10.4f}{k_used:>9.2f}{confidence:>7.3f}{runtime:>9.1f}")


def main():
    parser = argparse.ArgumentParser(description="Fixed-k majority voting experiment")

    parser.add_argument("--method", default="fixed_k_majority", choices=["fixed_k_majority"])
    parser.add_argument("--dataset-path", default="data/aime.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--k-values", default="1,10,100,1000")
    parser.add_argument("--subset-size", type=int, default=300)
    parser.add_argument("--subset-seed", type=int, default=123)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--early-stop-confidence", type=float, default=0.95)
    parser.add_argument("--early-stop-stability", type=int, default=50)
    parser.add_argument("--early-stop-min-samples", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=32, help="1 runs serially")
    parser.add_argument("--per-example-dir", default="experiments")

    run_fixed_k(parser.parse_args())


if __name__ == "__main__":
    main()
