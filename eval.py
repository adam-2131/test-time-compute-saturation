from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any

from utils import call_model
from parse import extract_final_answer
from prompting import format_math_prompt

# Debug flag - set to True to print budget info for first problem
DEBUG_BUDGET = False


def _derive_seed(base_seed: int, *, problem_index: int, call_index: int, tag: str) -> int:
    """
    Deterministic per-call seed derivation (stable across processes/runs).
    OpenAI expects an int; keep it in 32-bit signed range.
    """
    payload = f"{base_seed}:{problem_index}:{call_index}:{tag}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _majority_vote(answers: list[str | None]) -> str | None:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    # Deterministic tie-break: highest count, then lexicographic.
    best_count = max(counts.values())
    best = sorted([a for a, c in counts.items() if c == best_count])[0]
    return best


def run_best_of_n_with_budget(
    problems: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    budget_tokens: int,
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Uniform budgeted sampling: keep drawing independent samples until the decoded
    completion-token budget would be exhausted.

    Returns per-problem dicts with keys used by main.py:
      - id, pass_at_k, majority_correct
      - decoded_tokens, total_completion_tokens_used, total_tokens_used
      - total_calls, cap_hit_calls
    """
    results: list[dict[str, Any]] = []

    for p_idx, p in enumerate(problems):
        if DEBUG_BUDGET and p_idx == 0:
            print(f"[budget] method=uniform B={budget_tokens}")

        prompt = format_math_prompt(p["problem"])
        gold = str(p["answer"]).strip()

        decoded_tokens = 0
        total_completion_tokens_used = 0
        total_tokens_used = 0
        total_calls = 0
        cap_hit_calls = 0

        answers: list[str | None] = []
        pass_at_k = False

        call_i = 0
        while True:
            remaining = budget_tokens - decoded_tokens
            if remaining <= 0:
                break
            max_tokens_this = max(1, min(int(remaining), int(max_tokens)))

            out = call_model(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens_this,
                seed=_derive_seed(seed, problem_index=p_idx, call_index=call_i, tag="uniform"),
            )
            call_i += 1

            used_completion = int(out.completion_tokens or 0)
            used_total = int(out.total_tokens or (used_completion + int(out.prompt_tokens or 0)))

            decoded_tokens += used_completion
            total_completion_tokens_used += used_completion
            total_tokens_used += used_total
            total_calls += 1
            if out.completion_tokens is not None and used_completion >= max_tokens_this:
                cap_hit_calls += 1

            pred = extract_final_answer(out.text)
            answers.append(pred)
            if pred is not None and pred == gold:
                pass_at_k = True

            # Safety: if the API ever reports 0 completion tokens, don't loop forever.
            if used_completion <= 0:
                break

        majority = _majority_vote(answers)
        majority_correct = bool(majority is not None and majority == gold)

        results.append(
            {
                "id": p["id"],
                "pass_at_k": pass_at_k,
                "majority_correct": majority_correct,
                "decoded_tokens": decoded_tokens,
                "total_completion_tokens_used": total_completion_tokens_used,
                "total_tokens_used": total_tokens_used,
                "total_calls": total_calls,
                "cap_hit_calls": cap_hit_calls,
                "samples_taken": len(answers),
            }
        )

    return results


def run_two_phase_budgeted_sampling(
    problems: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens_cap: int,
    budget_tokens: int,
    *,
    N: int = 16,
    K: int = 4,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Two-phase budgeted sampling (2 rounds, no explicit trace continuation):
      - Round 1: N short calls to cheaply estimate a self-consistent answer.
      - Round 2: K longer calls, targeting the K "survivor" seeds from the most
        frequent Round-1 answer groups.
      - pass@k: any correct answer across *all* calls.
      - majority_correct: majority vote across Round-2 extracted answers (fallback to survivors).
    """
    results: list[dict[str, Any]] = []

    for p_idx, p in enumerate(problems):
        if DEBUG_BUDGET and p_idx == 0:
            print(f"[budget] method=two_phase B={budget_tokens} N={N} K={K}")

        prompt = format_math_prompt(p["problem"])
        gold = str(p["answer"]).strip()

        decoded_tokens = 0
        total_completion_tokens_used = 0
        total_tokens_used = 0
        total_calls = 0
        cap_hit_calls = 0

        pass_at_k = False

        # -------------------------
        # Round 1 (cheap scouting)
        # -------------------------
        round1_budget = max(1, int(0.25 * budget_tokens))
        round1_max_tokens = max(1, min(int(max_tokens_cap), round1_budget // max(1, int(N))))

        round1: list[dict[str, Any]] = []
        for i in range(int(N)):
            remaining = budget_tokens - decoded_tokens
            if remaining <= 0:
                break
            max_tokens_this = max(1, min(int(remaining), int(round1_max_tokens)))
            s = _derive_seed(seed, problem_index=p_idx, call_index=i, tag="two_phase_r1")
            out = call_model(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens_this,
                seed=s,
            )
            used_completion = int(out.completion_tokens or 0)
            used_total = int(out.total_tokens or (used_completion + int(out.prompt_tokens or 0)))

            decoded_tokens += used_completion
            total_completion_tokens_used += used_completion
            total_tokens_used += used_total
            total_calls += 1
            if out.completion_tokens is not None and used_completion >= max_tokens_this:
                cap_hit_calls += 1

            pred = extract_final_answer(out.text)
            if pred is not None and pred == gold:
                pass_at_k = True

            round1.append({"i": i, "seed": s, "answer": pred})

            if used_completion <= 0:
                break

        # Group answers by frequency and select survivors.
        counts = Counter(r["answer"] for r in round1 if r["answer"] is not None)
        # Highest frequency first; deterministic tie-break by answer string.
        ranked_answers = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
        top_answers = [a for a, _ in ranked_answers[: max(1, int(K))]] if ranked_answers else []

        survivors: list[dict[str, Any]] = []
        if top_answers:
            for r in round1:
                if r["answer"] in set(top_answers):
                    survivors.append(r)
                if len(survivors) >= int(K):
                    break
        else:
            survivors = round1[: int(K)]

        # -------------------------
        # Round 2 (spend remaining)
        # -------------------------
        r2_answers: list[str | None] = []
        if survivors:
            remaining = max(0, budget_tokens - decoded_tokens)
            per_call_budget = max(1, remaining // len(survivors)) if remaining else 1
            round2_max_tokens = max(1, min(int(max_tokens_cap), int(per_call_budget)))

            for j, s in enumerate(survivors):
                remaining = budget_tokens - decoded_tokens
                if remaining <= 0:
                    break
                max_tokens_this = max(1, min(int(remaining), int(round2_max_tokens)))

                out = call_model(
                    prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens_this,
                    seed=int(s["seed"]),
                )
                used_completion = int(out.completion_tokens or 0)
                used_total = int(out.total_tokens or (used_completion + int(out.prompt_tokens or 0)))

                decoded_tokens += used_completion
                total_completion_tokens_used += used_completion
                total_tokens_used += used_total
                total_calls += 1
                if out.completion_tokens is not None and used_completion >= max_tokens_this:
                    cap_hit_calls += 1

                pred = extract_final_answer(out.text)
                r2_answers.append(pred)
                if pred is not None and pred == gold:
                    pass_at_k = True

                if used_completion <= 0:
                    break

        if not r2_answers:
            r2_answers = [s.get("answer") for s in survivors]

        final = _majority_vote(r2_answers)
        majority_correct = bool(final is not None and final == gold)

        results.append(
            {
                "id": p["id"],
                "pass_at_k": pass_at_k,
                "majority_correct": majority_correct,
                "decoded_tokens": decoded_tokens,
                "total_completion_tokens_used": total_completion_tokens_used,
                "total_tokens_used": total_tokens_used,
                "total_calls": total_calls,
                "cap_hit_calls": cap_hit_calls,
                "samples_taken": len(round1) + len(r2_answers),
            }
        )

    return results


# Back-compat: old name used in earlier drafts.
run_successive_halving_with_budget = run_two_phase_budgeted_sampling


# =============================================================================
# Fixed-k Majority Voting (new experimental path per locked spec)
# =============================================================================

import math
import time


def majority_stats(answers: list[str | None]) -> tuple[str | None, float, float]:
    """
    Compute majority vote statistics from a list of extracted answers.

    Returns:
        (prediction, confidence, entropy)
        - prediction: majority answer (or None if no valid answers)
        - confidence: fraction of votes for majority answer (0.0 if no valid answers)
        - entropy: Shannon entropy of vote distribution in bits (0.0 if no valid answers)
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0.0, 0.0

    counts = Counter(valid)
    total = len(valid)

    # Majority prediction (deterministic tie-break: highest count, then lexicographic)
    best_count = max(counts.values())
    prediction = sorted([a for a, c in counts.items() if c == best_count])[0]

    # Confidence = fraction of votes for majority
    confidence = best_count / total

    # Shannon entropy in bits
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return prediction, confidence, entropy


def run_fixed_k_majority(
    problems: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    k_max: int,
    *,
    seed: int = 0,
    early_stop_stability: int = 50,
    early_stop_confidence: float = 0.95,
) -> list[dict[str, Any]]:
    """
    Fixed-k majority voting with mandatory early stopping.

    For each problem:
      - Sample up to k_max completions
      - Stop early if:
        (a) majority prediction unchanged for last `early_stop_stability` samples, OR
        (b) majority vote confidence >= `early_stop_confidence`
      - Compute final prediction via majority vote

    Args:
        problems: list of problem dicts with "id", "problem", "answer"
        model: model name
        temperature: sampling temperature
        max_tokens: max tokens per completion
        k_max: maximum number of samples per problem
        seed: base seed for deterministic sampling
        early_stop_stability: number of consecutive samples with unchanged majority to trigger stop
        early_stop_confidence: confidence threshold to trigger stop

    Returns:
        list of per-problem result dicts with:
          - example_id, k_max, k_used, prediction, confidence, entropy, correct, runtime_seconds
    """
    results: list[dict[str, Any]] = []

    for p_idx, p in enumerate(problems):
        start_time = time.perf_counter()

        prompt = format_math_prompt(p["problem"])
        gold = str(p["answer"]).strip()

        answers: list[str | None] = []
        k_used = 0

        # Track majority stability for early stopping
        prev_majority: str | None = None
        stability_counter = 0

        for call_i in range(k_max):
            # Generate one sample
            out = call_model(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=_derive_seed(seed, problem_index=p_idx, call_index=call_i, tag="fixed_k"),
            )

            pred = extract_final_answer(out.text)
            answers.append(pred)
            k_used += 1

            # Compute current majority stats
            current_majority, current_confidence, _ = majority_stats(answers)

            # Check early stopping: confidence threshold
            if current_confidence >= early_stop_confidence:
                break

            # Check early stopping: stability (majority unchanged for N consecutive samples)
            if current_majority == prev_majority and current_majority is not None:
                stability_counter += 1
                if stability_counter >= early_stop_stability:
                    break
            else:
                stability_counter = 1  # Reset: this sample is the first with new majority
                prev_majority = current_majority

        # Final aggregation
        prediction, confidence, entropy = majority_stats(answers)
        correct = bool(prediction is not None and prediction == gold)

        elapsed = time.perf_counter() - start_time

        results.append(
            {
                "example_id": p["id"],
                "k_max": k_max,
                "k_used": k_used,
                "prediction": prediction,
                "confidence": confidence,
                "entropy": entropy,
                "correct": correct,
                "runtime_seconds": elapsed,
                # Also include gold for debugging (not in spec CSV but useful internally)
                "_gold": gold,
            }
        )

    return results

