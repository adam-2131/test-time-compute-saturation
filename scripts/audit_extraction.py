#!/usr/bin/env python3
"""
Audit answer extraction correctness from *cached* model outputs.

This script does NOT change extraction logic and does NOT make new model calls.
It reconstructs cache keys (matching utils.call_model) to load cached outputs
and writes a CSV for manual reviewer annotation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from datetime import datetime
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data import load_frontier_math
from parse import extract_final_answer
from prompting import format_math_prompt


def _cache_key(
    *,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    cache_salt: int | None,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt": prompt,
        "seed": seed,
    }
    # Important: legacy behavior omits cache_salt entirely when None.
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_cached_call(
    *,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    cache_salt: int | None,
) -> dict[str, Any] | None:
    key = _cache_key(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        cache_salt=cache_salt,
    )
    path = os.path.join(".cache", f"{key}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    d["_cache_path"] = path
    return d


def _iter_uniform_cached_outputs(
    *,
    problem: dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
    budget_tokens: int,
    seed: int | None,
) -> list[dict[str, Any]]:
    prompt = format_math_prompt(problem["problem"])
    out_rows: list[dict[str, Any]] = []
    decoded = 0
    i = 0
    while True:
        cached = _load_cached_call(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            cache_salt=(i if temperature != 0.0 else None),
        )
        if cached is None:
            break

        ctoks = int(cached.get("completion_tokens") or 0)
        if decoded + ctoks > budget_tokens:
            break
        decoded += ctoks

        text = cached.get("text", "")
        out_rows.append(
            {
                "round": "uniform",
                "draw_index": i,
                "prompt": prompt,
                "raw_output": text,
                "extracted_answer": extract_final_answer(text),
                "gold_answer": problem.get("answer"),
                "completion_tokens": ctoks,
                "prompt_tokens": cached.get("prompt_tokens", ""),
                "total_tokens": cached.get("total_tokens", ""),
                "cache_path": cached.get("_cache_path", ""),
            }
        )
        i += 1
    return out_rows


def _iter_two_stage_sc_cached_outputs(
    *,
    problem: dict[str, Any],
    model: str,
    temperature: float,
    max_tokens_cap: int,
    budget_tokens: int,
    seed: int | None,
    N: int,
    K: int,
) -> list[dict[str, Any]]:
    prompt = format_math_prompt(problem["problem"])
    out_rows: list[dict[str, Any]] = []

    # Mirror eval.py logic (no behavior changes).
    round1_budget = int(0.25 * budget_tokens)
    round1_max_tokens = max(1, min(round1_budget // N, max_tokens_cap))

    spent = 0
    draw_idx = 0

    # Round 1 (always N calls; spent is tracked but not capped mid-round in eval.py).
    for i in range(N):
        cached = _load_cached_call(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=round1_max_tokens,
            seed=seed,
            cache_salt=(draw_idx if temperature != 0.0 else None),
        )
        if cached is None:
            return out_rows  # stop early if cache missing

        text = cached.get("text", "")
        ctoks = int(cached.get("completion_tokens") or 0)
        spent += ctoks
        out_rows.append(
            {
                "round": "round1",
                "draw_index": draw_idx,
                "prompt": prompt,
                "raw_output": text,
                "extracted_answer": extract_final_answer(text),
                "gold_answer": problem.get("answer"),
                "completion_tokens": ctoks,
                "prompt_tokens": cached.get("prompt_tokens", ""),
                "total_tokens": cached.get("total_tokens", ""),
                "cache_path": cached.get("_cache_path", ""),
            }
        )
        draw_idx += 1

    # Round 2 (batches of K, max_tokens=max_tokens_cap, with budget post-check).
    while spent < budget_tokens:
        remaining = budget_tokens - spent
        if remaining < K:
            break

        batch_spent = 0
        for _ in range(K):
            if spent + batch_spent >= budget_tokens:
                break

            cached = _load_cached_call(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens_cap,
                seed=seed,
                cache_salt=(draw_idx if temperature != 0.0 else None),
            )
            if cached is None:
                return out_rows

            text = cached.get("text", "")
            ctoks = int(cached.get("completion_tokens") or 0)
            if spent + batch_spent + ctoks > budget_tokens:
                break

            batch_spent += ctoks
            out_rows.append(
                {
                    "round": "round2",
                    "draw_index": draw_idx,
                    "prompt": prompt,
                    "raw_output": text,
                    "extracted_answer": extract_final_answer(text),
                    "gold_answer": problem.get("answer"),
                    "completion_tokens": ctoks,
                    "prompt_tokens": cached.get("prompt_tokens", ""),
                    "total_tokens": cached.get("total_tokens", ""),
                    "cache_path": cached.get("_cache_path", ""),
                }
            )
            draw_idx += 1

        spent += batch_spent
        if batch_spent == 0:
            break

    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", default="data/aime.jsonl")
    ap.add_argument(
        "--method",
        default="uniform_best_of_n_budgeted",
        choices=["uniform_best_of_n_budgeted", "two_stage_self_consistency", "successive_halving_budgeted"],
        help="Method used to generate cached outputs (successive_halving_budgeted is an alias).",
    )
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048, help="max_tokens used in generation (uniform / stage-2 cap)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compute-multiplier", type=int, default=10)
    ap.add_argument("--base-budget-tokens", type=int, default=1004)
    ap.add_argument("--limit-problems", type=int, default=None)
    ap.add_argument("--n", type=int, default=50, help="Number of cached outputs to sample")
    ap.add_argument("--rng-seed", type=int, default=0, help="Seed for which outputs are selected")
    ap.add_argument("--out", default=None, help="Output CSV path (default: results/extraction_audit_<timestamp>.csv)")
    ap.add_argument("--N", type=int, default=16, help="two_stage_self_consistency: round-1 sample count")
    ap.add_argument("--K", type=int, default=4, help="two_stage_self_consistency: round-2 batch size")
    args = ap.parse_args()

    method = args.method
    if method == "successive_halving_budgeted":
        method = "two_stage_self_consistency"

    budget_tokens = args.compute_multiplier * args.base_budget_tokens
    problems = load_frontier_math(args.dataset_path)
    if args.limit_problems is not None:
        problems = problems[: args.limit_problems]

    all_outputs: list[dict[str, Any]] = []
    for p in problems:
        if method == "uniform_best_of_n_budgeted":
            outs = _iter_uniform_cached_outputs(
                problem=p,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                budget_tokens=budget_tokens,
                seed=args.seed,
            )
        else:
            outs = _iter_two_stage_sc_cached_outputs(
                problem=p,
                model=args.model,
                temperature=args.temperature,
                max_tokens_cap=args.max_tokens,
                budget_tokens=budget_tokens,
                seed=args.seed,
                N=args.N,
                K=args.K,
            )
        for o in outs:
            o["problem_id"] = p.get("id")
            o["problem"] = p.get("problem")
            o["method"] = method
            o["compute_multiplier"] = args.compute_multiplier
            o["budget_tokens"] = budget_tokens
        all_outputs.extend(outs)

    if not all_outputs:
        raise RuntimeError(
            "No cached outputs found. Ensure `.cache/` exists and you have previously run the experiment "
            "with matching (model, temperature, max_tokens, seed, method, multiplier)."
        )

    rng = random.Random(args.rng_seed)
    rng.shuffle(all_outputs)
    sampled = all_outputs[: min(args.n, len(all_outputs))]

    os.makedirs("results", exist_ok=True)
    if args.out:
        out_path = args.out
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join("results", f"extraction_audit_{ts}.csv")

    fieldnames = [
        "method",
        "compute_multiplier",
        "budget_tokens",
        "problem_id",
        "gold_answer",
        "extracted_answer",
        "round",
        "draw_index",
        "completion_tokens",
        "prompt_tokens",
        "total_tokens",
        "cache_path",
        "problem",
        "prompt",
        "raw_output",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sampled:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"wrote {out_path} ({len(sampled)}/{len(all_outputs)} cached outputs sampled)")


if __name__ == "__main__":
    main()


