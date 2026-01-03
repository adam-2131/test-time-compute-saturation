# main.py
import argparse
import json
import os
import random
from pathlib import Path

from data import load_frontier_math
from eval import run_best_of_n_with_budget, run_successive_halving_with_budget
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

# Two-stage self-consistency hyperparameters (fixed for now)
SH_N = 16  # initial samples
SH_K = 4   # survivors


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


def run(*, dataset_path: str, dataset_name: str, limit: int | None, multipliers: list[int], method: str, seed: int, debug_budget: bool = False):
    problems = load_frontier_math(dataset_path)
    # Backward-compatible alias: this implementation is NOT trace-continuation successive halving.
    # It is a two-stage self-consistency style procedure (round-1 vote to select survivors; round-2 resampling + vote).
    if method == "successive_halving_budgeted":
        method = "two_stage_self_consistency"

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
                MODEL,
                TEMPERATURE,
                MAX_TOKENS,
                budget_tokens,
                seed=seed,
            )
        elif method == "two_stage_self_consistency":
            results = run_successive_halving_with_budget(
                problems,
                MODEL,
                TEMPERATURE,
                MAX_TOKENS,
                budget_tokens,
                N=SH_N,
                K=SH_K,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        accuracy = sum(1 for r in results if r["pass"]) / len(results)
        pass_at_k_accuracy = sum(1 for r in results if r.get("pass_at_k", r["pass"])) / len(results)
        majority_vote_accuracy = sum(1 for r in results if r.get("majority_correct", r["pass"])) / len(results)
        total_decoded_tokens_used = sum(r["decoded_tokens"] for r in results)
        total_completion_tokens_used = sum(r.get("total_completion_tokens_used", r["decoded_tokens"]) for r in results)
        total_tokens_used = sum(r.get("total_tokens_used", 0) for r in results)
        total_calls = sum(r.get("total_calls", 0) for r in results)
        cap_hit_calls = sum(r.get("cap_hit_calls", 0) for r in results)
        cap_hit_rate = (cap_hit_calls / total_calls) if total_calls else 0.0
        rows.append((m, accuracy, pass_at_k_accuracy, majority_vote_accuracy, cap_hit_rate, total_decoded_tokens_used))

        # Enforce semantic separation:
        # - sanity_check ONLY for toy/debug runs with num_problems < 10
        # - subset= ONLY for real aime/frontier_math runs with num_problems >= 10
        num_problems = len(problems)
        is_toy = dataset_name == "toy"
        is_real_dataset = dataset_name in {"aime", "frontier_math"}

        if is_toy and num_problems < 10:
            # Toy/debug run
            notes = "sanity_check"
        elif is_real_dataset and num_problems >= 10 and limit is not None:
            # Real run with subset
            notes = f"subset={limit}"
        elif is_real_dataset and num_problems >= 10:
            # Real run without subset specification
            notes = f"subset={num_problems}"
        else:
            # Fallback: sanity_check for small runs, empty for ambiguous cases
            if num_problems < 10:
                notes = "sanity_check"
            else:
                notes = f"subset={num_problems}"

        # Add seed to notes for all methods
        notes = f"{notes} seed={seed}"
        
        # For two_stage_self_consistency, add additional hyperparameters to notes
        if method == "two_stage_self_consistency":
            notes = f"{notes} rounds=2 N={SH_N} K={SH_K}"

        # Diagnostics: how often completions hit the max_tokens cap (per-call, across the run).
        notes = f"{notes} cap_hit_rate={cap_hit_rate:.4f} cap_hits={cap_hit_calls} calls={total_calls}"

        append_result_row(
            model=MODEL,
            dataset=dataset_name,
            method=method,
            compute_multiplier=m,
            accuracy=accuracy,
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
            "model": MODEL,
            "dataset": dataset_name,
            "method": method,
            "seed": seed,
            "compute_multiplier": m,
            "num_problems": len(results),
            # Method-dependent correctness (matches CSV "accuracy" semantics).
            "per_problem_correct": [bool(r.get("pass", False)) for r in results],
            # Also store both metrics for optional analysis.
            "per_problem_pass_at_k": [bool(r.get("pass_at_k", r.get("pass", False))) for r in results],
            "per_problem_majority_correct": [bool(r.get("majority_correct", r.get("pass", False))) for r in results],
        }
        with open(per_problem_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(per_problem, ensure_ascii=False) + "\n")

    header = (
        f"{'multiplier':>10}  {'accuracy':>10}  {'pass@k':>10}  {'majority':>10}  {'cap_hit':>10}  "
        f"{'total_decoded_tokens_used':>25}"
    )
    print(f"Method: {method}")
    print(header)
    print("-" * len(header))
    # Note: 'accuracy' remains method-dependent (uniform=pass@k, successive=majority).
    for m, acc, pass_k, maj, cap_hit, toks in rows:
        print(f"{m:>10}  {acc:>10.4f}  {pass_k:>10.4f}  {maj:>10.4f}  {cap_hit:>10.4f}  {toks:>25}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--multipliers", default=",".join(map(str, DEFAULT_MULTIPLIERS)))
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=["uniform_best_of_n_budgeted", "two_stage_self_consistency", "successive_halving_budgeted"],
        help="Evaluation method to use (successive_halving_budgeted is an alias)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility (controls model sampling)",
    )
    parser.add_argument(
        "--debug-budget",
        action="store_true",
        default=False,
        help="Print budget B for each multiplier (for debugging budget consistency)",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name or _infer_dataset_name(args.dataset_path)
    multipliers = _parse_multipliers(args.multipliers)
    run(
        dataset_path=args.dataset_path,
        dataset_name=dataset_name,
        limit=args.limit,
        multipliers=multipliers,
        method=args.method,
        seed=args.seed,
        debug_budget=args.debug_budget,
    )
