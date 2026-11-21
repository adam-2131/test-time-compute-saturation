
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

COLUMNS = [
    "example_id",
    "k_max",
    "k_used",
    "valid_votes",
    "invalid_votes",
    "token_limit_votes",
    "prediction",
    "confidence",
    "entropy",
    "correct",
    "completion_tokens",
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
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=COLUMNS)
        writer.writeheader()

        for result in results:
            writer.writerow({
                **{column: result[column] for column in COLUMNS},
                "prediction": "" if result["prediction"] is None else result["prediction"],
                "confidence": f"{result['confidence']:.6f}",
                "entropy": f"{result['entropy']:.6f}",
                "correct": int(result["correct"]),
                "runtime_seconds": f"{result['runtime_seconds']:.4f}",
            })


def run_fixed_k(args):
    problems = select_subset(
        load_jsonl_dataset(args.dataset_path),
        args.subset_size,
        args.subset_seed,
    )

    # The first N problems of the subset keep their position, so their seeds
    # and cache entries are the same as in the full run.
    if args.limit is not None:
        problems = problems[:args.limit]

    if not problems:
        raise ValueError(f"No problems loaded from {args.dataset_path}")

    k_values = parse_int_list(args.k_values)

    out_dir = Path(args.per_example_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fixed-k plurality voting on {Path(args.dataset_path).stem}, model {args.model}")
    print(
        f"{len(problems)} problems (subset_size={args.subset_size}, "
        f"subset_seed={args.subset_seed}), k = {k_values}"
    )
    print(f"temperature {TEMPERATURE}, max_tokens {args.max_tokens}, seed {args.seed}")
    print(
        f"Early stop: confidence {args.early_stop_confidence}, "
        f"stability {args.early_stop_stability}, "
        f"minimum samples {args.early_stop_min_samples}, "
        f"minimum valid answers {args.early_stop_min_valid}\n"
    )

    summary = []

    for k in k_values:
        print(f"k_max={k}")

        results = run_fixed_k_majority(
            problems,
            args.model,
            TEMPERATURE,
            args.max_tokens,
            k_max=k,
            seed=args.seed,
            early_stop_stability=args.early_stop_stability,
            early_stop_confidence=args.early_stop_confidence,
            early_stop_min_samples=args.early_stop_min_samples,
            early_stop_min_valid=args.early_stop_min_valid,
            max_workers=args.max_workers,
            progress=True,
        )

        output_path = out_dir / f"k_{k}.csv"
        write_csv(output_path, results)
        print(f"Wrote {output_path} ({len(results)} rows)")

        drawn = sum(result["k_used"] for result in results)
        summary.append((
            k,
            fmean(result["correct"] for result in results),
            fmean(result["k_used"] for result in results),
            sum(result["valid_votes"] for result in results) / drawn,
            sum(result["token_limit_votes"] for result in results) / drawn,
            fmean(result["confidence"] for result in results),
        ))

    print(
        f"\n{'k_max':>6}{'accuracy':>10}{'k_used':>9}"
        f"{'valid':>8}{'cut off':>9}{'confidence':>12}"
    )

    for k, accuracy, k_used, valid, cut_off, confidence in summary:
        print(
            f"{k:>6}{accuracy:>10.4f}{k_used:>9.2f}"
            f"{valid:>8.1%}{cut_off:>9.1%}{confidence:>12.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-k sampling with plurality aggregation"
    )

    parser.add_argument(
        "--method",
        default="fixed_k_majority",
        choices=["fixed_k_majority"],
    )
    parser.add_argument("--dataset-path", default="data/aime.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="output token limit per response")
    parser.add_argument("--k-values", default="1,10,100,1000")
    parser.add_argument("--subset-size", type=int, default=300)
    parser.add_argument("--subset-seed", type=int, default=123)
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N problems of the subset (for pilot runs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--early-stop-confidence", type=float, default=0.95)
    parser.add_argument("--early-stop-stability", type=int, default=50)
    parser.add_argument("--early-stop-min-samples", type=int, default=10)
    parser.add_argument("--early-stop-min-valid", type=int, default=10,
                        help="valid answers needed before either stopping rule applies")
    parser.add_argument("--max-workers", type=int, default=32, help="1 runs serially")
    parser.add_argument("--per-example-dir", default="experiments")

    run_fixed_k(parser.parse_args())


if __name__ == "__main__":
    main()
