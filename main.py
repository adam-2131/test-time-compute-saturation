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

from data import load_jsonl_dataset
from eval import run_best_of_n_with_budget, run_two_phase_budgeted_sampling
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


def run(*, dataset_path: str, dataset_name: str, limit: int | None, multipliers: list[int], method: str, seed: int, debug_budget: bool = False):
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
                MODEL,
                TEMPERATURE,
                MAX_TOKENS,
                budget_tokens,
                seed=seed,
            )
        elif method == "two_phase_budgeted_sampling":
            results = run_two_phase_budgeted_sampling(
                problems,
                MODEL,
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
            model=MODEL,
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
            "model": MODEL,
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
        "--debug-budget",
        action="store_true",
        default=False,
        help="Print budget B for each multiplier (for debugging budget consistency)",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name or _infer_dataset_name(args.dataset_path)
    multipliers = _parse_multipliers(args.multipliers)
    try:
        run(
            dataset_path=args.dataset_path,
            dataset_name=dataset_name,
            limit=args.limit,
            multipliers=multipliers,
            method=args.method,
            seed=args.seed,
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
