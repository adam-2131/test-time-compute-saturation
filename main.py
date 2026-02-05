# main.py
"""
Main experiment runner for test-time compute scaling.

Runs budgeted evaluation methods on math problems and logs results.

Primary metric: pass_at_k_accuracy (fraction of problems with at least one correct sample)
Compute unit: decoded completion tokens
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import csv

from data import load_jsonl_dataset
from eval import run_best_of_n_with_budget, run_two_phase_budgeted_sampling, run_fixed_k_majority
from results_logger import append_result_row, get_run_id

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7
MAX_TOKENS = 2048
DEFAULT_DATASET_PATH = "data/aime.jsonl"
DEFAULT_MULTIPLIERS = [1, 10, 100]
DEFAULT_METHOD = "uniform_best_of_n_budgeted"
DEFAULT_SEED = 0

# Fixed per-problem token budget baseline (deterministic, no API jitter)
# For multiplier m, budget_tokens = m * BASE_BUDGET_TOKENS
BASE_BUDGET_TOKENS = 1004

# Two-phase budgeted sampling hyperparameters (fixed)
TWO_PHASE_N = 16  # initial samples in round 1
TWO_PHASE_K = 4   # batch size in round 2

# Fixed-k majority voting defaults (new experimental path)
DEFAULT_K_VALUES = [1, 10, 100, 1000]
DEFAULT_SUBSET_SIZE = 300
DEFAULT_SUBSET_SEED = 123
DEFAULT_EARLY_STOP_STABILITY = 50
DEFAULT_EARLY_STOP_CONFIDENCE = 0.95
DEFAULT_PER_EXAMPLE_DIR = "experiments"

# Track if we've already printed alias warnings
_ALIAS_WARNINGS_PRINTED: set[str] = set()

# CLI alias -> canonical method name mapping
METHOD_ALIASES = {
    "successive_halving_budgeted": "two_phase_budgeted_sampling",
    "two_stage_self_consistency": "two_phase_budgeted_sampling",
}


def _infer_dataset_name(dataset_path: str) -> str:
    return Path(dataset_path).stem or "dataset"


def _parse_multipliers(s: str) -> list[int]:
    out: list[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_k_values(s: str) -> list[int]:
    """Parse comma-separated k values for fixed-k method."""
    out: list[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _select_subset(
    problems: list[dict],
    subset_size: int,
    subset_seed: int,
) -> list[dict]:
    """
    Deterministic seeded subset selection for fixed-k experiments.

    If len(problems) <= subset_size, returns all problems (order preserved).
    Otherwise, uses seeded shuffle to select exactly subset_size problems.
    """
    if len(problems) <= subset_size:
        return problems

    # Deterministic shuffle using subset_seed
    rng = random.Random(subset_seed)
    indices = list(range(len(problems)))
    rng.shuffle(indices)
    selected_indices = sorted(indices[:subset_size])  # Keep original order for selected
    return [problems[i] for i in selected_indices]


def run_fixed_k(
    *,
    dataset_path: str,
    dataset_name: str,
    k_values: list[int],
    subset_size: int,
    subset_seed: int,
    early_stop_stability: int,
    early_stop_confidence: float,
    per_example_dir: str,
    seed: int,
    model: str = MODEL,
):
    """
    Run fixed-k majority voting experiments (new experimental path).

    For each k in k_values:
      - Sample up to k completions per problem (with early stopping)
      - Write per-example CSV to {per_example_dir}/k_{k}.csv

    Uses deterministic seeded subset selection; same subset across all k.
    """
    # Load and select subset (deterministic, reused across all k)
    all_problems = load_jsonl_dataset(dataset_path)
    problems = _select_subset(all_problems, subset_size, subset_seed)

    if not problems:
        raise ValueError(f"No problems loaded from {dataset_path}")

    print(f"Fixed-k majority voting experiment")
    print(f"  Dataset: {dataset_name} ({len(problems)} problems, subset_seed={subset_seed})")
    print(f"  k values: {k_values}")
    print(f"  Early stop: stability={early_stop_stability}, confidence={early_stop_confidence}")
    print(f"  Output dir: {per_example_dir}/")
    print()

    os.makedirs(per_example_dir, exist_ok=True)

    # CSV columns per spec
    csv_columns = [
        "example_id",
        "k_max",
        "k_used",
        "prediction",
        "confidence",
        "entropy",
        "correct",
        "runtime_seconds",
    ]

    summary_rows: list[tuple[int, float, float, float, float]] = []

    for k in k_values:
        print(f"Running k_max={k} ...")

        results = run_fixed_k_majority(
            problems,
            model,
            TEMPERATURE,
            MAX_TOKENS,
            k_max=k,
            seed=seed,
            early_stop_stability=early_stop_stability,
            early_stop_confidence=early_stop_confidence,
        )

        # Write per-example CSV
        csv_path = os.path.join(per_example_dir, f"k_{k}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "example_id": r["example_id"],
                    "k_max": r["k_max"],
                    "k_used": r["k_used"],
                    "prediction": r["prediction"] if r["prediction"] is not None else "",
                    "confidence": f"{r['confidence']:.6f}",
                    "entropy": f"{r['entropy']:.6f}",
                    "correct": int(r["correct"]),
                    "runtime_seconds": f"{r['runtime_seconds']:.4f}",
                })

        # Compute summary stats
        accuracy = sum(1 for r in results if r["correct"]) / len(results)
        avg_k_used = sum(r["k_used"] for r in results) / len(results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_runtime = sum(r["runtime_seconds"] for r in results) / len(results)

        summary_rows.append((k, accuracy, avg_k_used, avg_confidence, avg_runtime))
        print(f"  -> {csv_path} ({len(results)} rows)")

    # Print summary table
    print()
    print("=" * 80)
    print("Fixed-k Majority Voting Summary")
    print("=" * 80)
    header = f"{'k_max':>8}  {'accuracy':>10}  {'avg_k_used':>12}  {'avg_conf':>10}  {'avg_runtime':>12}"
    print(header)
    print("-" * len(header))
    for k, acc, avg_k, avg_conf, avg_rt in summary_rows:
        print(f"{k:>8}  {acc:>10.4f}  {avg_k:>12.2f}  {avg_conf:>10.4f}  {avg_rt:>12.4f}s")

    print()
    print(f"Per-example CSVs written to: {per_example_dir}/")


def _resolve_method(method: str) -> str:
    """Resolve CLI method alias to canonical name, printing warning if alias used."""
    global _ALIAS_WARNINGS_PRINTED
    
    if method in METHOD_ALIASES:
        canonical = METHOD_ALIASES[method]
        if method not in _ALIAS_WARNINGS_PRINTED:
            print(
                f"NOTE: '{method}' is an alias for '{canonical}' "
                f"(two-phase budget schedule, no trace continuation).",
                file=sys.stderr
            )
            _ALIAS_WARNINGS_PRINTED.add(method)
        return canonical
    return method


def run(
    *,
    dataset_path: str,
    dataset_name: str,
    limit: int | None,
    multipliers: list[int],
    method: str,
    seed: int,
    model: str = MODEL,
    debug_budget: bool = False,
):
    problems = load_jsonl_dataset(dataset_path)
    
    # Resolve CLI alias to canonical method name
    method = _resolve_method(method)

    if limit is not None:
        problems = problems[:limit]
    if not problems:
        raise ValueError(f"No problems loaded from {dataset_path}")

    # Set random seeds for reproducibility (only affects model sampling, not budgets)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass  # numpy not required

    # Get run_id once for the entire sweep (all multipliers share same run_id)
    run_id = get_run_id()
    os.makedirs("results", exist_ok=True)
    per_problem_path = os.path.join("results", "per_problem_results.jsonl")

    rows = []

    for m in multipliers:
        # Budget is "decoded completion tokens" (ints); fixed ex ante.
        # SINGLE SOURCE OF TRUTH: budget_tokens = multiplier * BASE_BUDGET_TOKENS
        # This guarantees identical budgets across methods, seeds, and runs.
        budget_tokens = m * BASE_BUDGET_TOKENS
        
        if debug_budget:
            subset_str = f"subset={limit}" if limit else f"subset={len(problems)}"
            print(f"[budget] dataset={dataset_name} {subset_str} mult={m} B={budget_tokens}")

        if method == "uniform_best_of_n_budgeted":
            results = run_best_of_n_with_budget(
                problems,
                model,
                TEMPERATURE,
                MAX_TOKENS,
                budget_tokens,
                seed=seed,
            )
        elif method == "two_phase_budgeted_sampling":
            results = run_two_phase_budgeted_sampling(
                problems,
                model,
                TEMPERATURE,
                MAX_TOKENS,
                budget_tokens,
                N=TWO_PHASE_N,
                K=TWO_PHASE_K,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # PRIMARY METRIC: pass@k (comparable across all methods)
        pass_at_k_accuracy = sum(1 for r in results if r["pass_at_k"]) / len(results)
        
        # SECONDARY METRIC: majority vote accuracy
        majority_vote_accuracy = sum(1 for r in results if r.get("majority_correct", False)) / len(results)
        
        total_decoded_tokens_used = sum(r["decoded_tokens"] for r in results)
        total_completion_tokens_used = sum(r.get("total_completion_tokens_used", r["decoded_tokens"]) for r in results)
        total_tokens_used = sum(r.get("total_tokens_used", 0) for r in results)
        total_calls = sum(r.get("total_calls", 0) for r in results)
        cap_hit_calls = sum(r.get("cap_hit_calls", 0) for r in results)
        cap_hit_rate = (cap_hit_calls / total_calls) if total_calls else 0.0
        rows.append((m, pass_at_k_accuracy, majority_vote_accuracy, cap_hit_rate, total_decoded_tokens_used))

        # Build notes string
        num_problems = len(problems)
        is_toy = dataset_name == "toy"
        is_real_dataset = dataset_name in {"aime", "frontier_math"}

        if is_toy and num_problems < 10:
            notes = "sanity_check"
        elif is_real_dataset and num_problems >= 10 and limit is not None:
            notes = f"subset={limit}"
        elif is_real_dataset and num_problems >= 10:
            notes = f"subset={num_problems}"
        else:
            if num_problems < 10:
                notes = "sanity_check"
            else:
                notes = f"subset={num_problems}"

        # Add seed to notes for all methods
        notes = f"{notes} seed={seed}"
        
        # For two_phase_budgeted_sampling, add additional hyperparameters to notes
        if method == "two_phase_budgeted_sampling":
            notes = f"{notes} rounds=2 N={TWO_PHASE_N} K={TWO_PHASE_K}"

        # Diagnostics: cap hit rate
        notes = f"{notes} cap_hit_rate={cap_hit_rate:.4f} cap_hits={cap_hit_calls} calls={total_calls}"

        append_result_row(
            model=model,
            dataset=dataset_name,
            method=method,
            compute_multiplier=m,
            pass_at_k_accuracy=pass_at_k_accuracy,
            majority_vote_accuracy=majority_vote_accuracy,
            total_decoded_tokens=total_decoded_tokens_used,
            total_completion_tokens_used=total_completion_tokens_used,
            total_tokens_used=total_tokens_used,
            num_problems=len(problems),
            notes=notes,
            run_id=run_id,
        )

        # Per-problem artifact for post-hoc stats (no new model calls).
        # One JSONL line per (run_id, multiplier).
        per_problem = {
            "run_id": run_id,
            "model": model,
            "dataset": dataset_name,
            "method": method,
            "seed": seed,
            "compute_multiplier": m,
            "num_problems": len(results),
            # Primary metric: pass@k
            "per_problem_correct": [bool(r.get("pass_at_k", False)) for r in results],
            "per_problem_pass_at_k": [bool(r.get("pass_at_k", False)) for r in results],
            "per_problem_majority_correct": [bool(r.get("majority_correct", False)) for r in results],
        }
        with open(per_problem_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(per_problem, ensure_ascii=False) + "\n")

    # Print results table
    header = (
        f"{'multiplier':>10}  {'pass@k':>10}  {'majority':>10}  {'cap_hit':>10}  "
        f"{'total_decoded_tokens':>22}"
    )
    print(f"\nMethod: {method}")
    print(f"Primary metric: pass_at_k_accuracy")
    print(header)
    print("-" * len(header))
    for m, pass_k, maj, cap_hit, toks in rows:
        print(f"{m:>10}  {pass_k:>10.4f}  {maj:>10.4f}  {cap_hit:>10.4f}  {toks:>22}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run test-time compute scaling experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  uniform_best_of_n_budgeted     Uniform sampling until budget exhausted
  two_phase_budgeted_sampling    Two-phase budget schedule (no trace continuation)
  fixed_k_majority               Fixed-k parallel sampling with majority vote (new)
  
Aliases (deprecated):
  successive_halving_budgeted    -> two_phase_budgeted_sampling
  two_stage_self_consistency     -> two_phase_budgeted_sampling
"""
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--multipliers", default=",".join(map(str, DEFAULT_MULTIPLIERS)))
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=[
            "uniform_best_of_n_budgeted",
            "two_phase_budgeted_sampling",
            "fixed_k_majority",
            # Legacy aliases (will emit warnings):
            "two_stage_self_consistency",
            "successive_halving_budgeted",
        ],
        help="Evaluation method (legacy aliases emit warnings)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility (controls model sampling via derived per-draw seeds)",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help="OpenAI model name (override without editing code).",
    )
    parser.add_argument(
        "--debug-budget",
        action="store_true",
        default=False,
        help="Print budget B for each multiplier (for debugging budget consistency)",
    )
    # Fixed-k majority voting flags (only used when --method=fixed_k_majority)
    parser.add_argument(
        "--k-values",
        default=",".join(map(str, DEFAULT_K_VALUES)),
        help="Comma-separated k values for fixed_k_majority (default: 1,10,100,1000)",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=DEFAULT_SUBSET_SIZE,
        help="Subset size for fixed_k_majority (default: 300)",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=DEFAULT_SUBSET_SEED,
        help="Seed for deterministic subset selection in fixed_k_majority (default: 123)",
    )
    parser.add_argument(
        "--early-stop-stability",
        type=int,
        default=DEFAULT_EARLY_STOP_STABILITY,
        help="Stop if majority unchanged for this many samples (default: 50)",
    )
    parser.add_argument(
        "--early-stop-confidence",
        type=float,
        default=DEFAULT_EARLY_STOP_CONFIDENCE,
        help="Stop if confidence >= this threshold (default: 0.95)",
    )
    parser.add_argument(
        "--per-example-dir",
        default=DEFAULT_PER_EXAMPLE_DIR,
        help="Output directory for per-example CSVs (default: experiments)",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name or _infer_dataset_name(args.dataset_path)

    # Route to the appropriate experiment path
    if args.method == "fixed_k_majority":
        # New fixed-k experimental path (does not use multipliers/limit/budget)
        k_values = _parse_k_values(args.k_values)
        try:
            run_fixed_k(
                dataset_path=args.dataset_path,
                dataset_name=dataset_name,
                k_values=k_values,
                subset_size=args.subset_size,
                subset_seed=args.subset_seed,
                early_stop_stability=args.early_stop_stability,
                early_stop_confidence=args.early_stop_confidence,
                per_example_dir=args.per_example_dir,
                seed=args.seed,
                model=args.model,
            )
        except RuntimeError as e:
            msg = str(e)
            print(f"\nERROR: {msg}\n", file=sys.stderr)
            print(
                "If this is a timeout, try:\n"
                "  $env:OPENAI_REQUEST_TIMEOUT_S = \"120\"\n"
                "  $env:OPENAI_MAX_ATTEMPTS = \"8\"\n"
                "then re-run.\n",
                file=sys.stderr,
            )
            raise SystemExit(2)
    else:
        # Existing token-budgeted methods
        multipliers = _parse_multipliers(args.multipliers)
        try:
            run(
                dataset_path=args.dataset_path,
                dataset_name=dataset_name,
                limit=args.limit,
                multipliers=multipliers,
                method=args.method,
                seed=args.seed,
                model=args.model,
                debug_budget=args.debug_budget,
            )
        except RuntimeError as e:
            # Make network failures (timeouts/proxy) reviewer-friendly and actionable.
            msg = str(e)
            print(f"\nERROR: {msg}\n", file=sys.stderr)
            print(
                "If this is a timeout, try:\n"
                "  $env:OPENAI_REQUEST_TIMEOUT_S = \"120\"\n"
                "  $env:OPENAI_MAX_ATTEMPTS = \"8\"\n"
                "then re-run.\n",
                file=sys.stderr,
            )
            raise SystemExit(2)
