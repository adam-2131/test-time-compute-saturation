"""
Evaluation methods for test-time compute scaling experiments.

Methods:
  - run_best_of_n_with_budget: Uniform sampling until budget exhausted (pass@k primary)
  - run_two_phase_budgeted_sampling: Two-phase budget schedule (NOT successive halving)

IMPORTANT: "two_phase_budgeted_sampling" is a budget schedule, NOT adaptive allocation.
  - Round 1: short samples, scored by self-consistency
  - Round 2: fresh samples (no trace continuation), majority vote
  - Does NOT condition round 2 sampling on round 1 survivors' traces
"""
import hashlib
from collections import Counter

from utils import call_model
from parse import extract_final_answer
from prompting import format_math_prompt


def derive_call_seed(base_seed: int | None, problem_id: str, draw_idx: int) -> int | None:
    """
    Derive a deterministic seed for each model call to ensure sampling independence.
    
    Uses sha256 mixing (not Python hash()) for stability across runs and platforms.
    
    Args:
        base_seed: The base seed from the experiment config. If None, returns None.
        problem_id: Unique identifier for the problem (ensures different problems get different seeds)
        draw_idx: Index of this sample within the problem (ensures each draw gets a different seed)
    
    Returns:
        A 32-bit integer derived from the inputs, or None if base_seed is None.
    """
    if base_seed is None:
        return None
    s = f"{base_seed}|{problem_id}|{draw_idx}".encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return int(h[:8], 16)  # 32-bit int


def run_best_of_n_with_budget(problems, model, temperature, max_tokens, budget_tokens, *, seed=None):
    """
    Uniform sampling (best-of-N) under a decoded-token budget.
    
    Primary metric: pass@k (any correct among accepted samples)
    Secondary metric: majority_correct (majority vote over valid answers)
    
    Args:
        problems: list of problem dicts with "id", "problem", "answer"
        model: model name
        temperature: sampling temperature
        max_tokens: max tokens per completion
        budget_tokens: per-problem decoded token budget
        seed: base seed for deterministic sampling (derived per draw)
    
    Returns:
        list of result dicts
    """
    results = []

    for p in problems:
        prompt = format_math_prompt(p["problem"])
        problem_id = str(p["id"])
        samples_taken = 0
        decoded_tokens = 0
        total_completion_tokens_used = 0
        total_tokens_used = 0
        passed = False  # pass@k (any correct among accepted samples)
        answers: list[str | None] = []
        total_calls = 0
        cap_hit_calls = 0
        draw_idx = 0

        while True:
            if decoded_tokens >= budget_tokens:
                break

            # Derive unique seed for this specific call
            call_seed = derive_call_seed(seed, problem_id, draw_idx)

            out = call_model(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=call_seed,
                cache_salt=(draw_idx if temperature != 0.0 else None),
            )
            total_calls += 1
            draw_idx += 1

            ctoks = out.completion_tokens or 0
            ptoks = out.prompt_tokens or 0
            ttoks = out.total_tokens
            if ttoks is None:
                ttoks = ptoks + ctoks
            if ctoks == max_tokens:
                cap_hit_calls += 1

            # Preserve existing budgeting logic: budget is completion tokens only.
            if decoded_tokens + ctoks > budget_tokens:
                break

            decoded_tokens += ctoks
            total_completion_tokens_used += ctoks
            total_tokens_used += ttoks
            samples_taken += 1

            pred = extract_final_answer(out.text)
            answers.append(pred)
            if pred == p["answer"]:
                passed = True

        valid_answers = [a for a in answers if a is not None]
        majority_correct = False
        if valid_answers:
            majority_answer = Counter(valid_answers).most_common(1)[0][0]
            majority_correct = (majority_answer == p["answer"])

        # Primary metric is pass@k for ALL methods (comparable across methods)
        results.append(
            {
                "id": p["id"],
                "samples_taken": samples_taken,
                "decoded_tokens": decoded_tokens,
                "total_completion_tokens_used": total_completion_tokens_used,
                "total_tokens_used": total_tokens_used,
                "pass": passed,  # PRIMARY: pass@k
                "pass_at_k": passed,
                "majority_correct": majority_correct,  # SECONDARY
                "total_calls": total_calls,
                "cap_hit_calls": cap_hit_calls,
            }
        )

    return results


# =============================================================================
# Two-phase budgeted sampling (NOT successive halving / NOT adaptive allocation)
#   R=2 rounds, fixed N=16, K=4
# =============================================================================
# Algorithm:
#   Round 1: generate N samples with short max_tokens = floor(0.25*B / N)
#            Score each by answer frequency (self-consistency).
#            Select K survivors from most frequent answer groups.
#   Round 2: generate (possibly multiple batches of) K fresh samples with max_tokens = max_tokens_cap
#            Final answer = majority vote across round-2 answers.
#
# IMPORTANT: This is a budget SCHEDULE, not adaptive allocation:
#   1. Round 2 does NOT continue from round 1 traces (fresh samples)
#   2. There is no actual trace continuation or "halving" of candidates
#   3. Survivor selection only affects which answer groups are "favored" in round 1 scoring


def run_two_phase_budgeted_sampling(
    problems, model, temperature, max_tokens_cap, budget_tokens, *, N=16, K=4, seed=None
):
    """
    Two-phase budgeted sampling with R=2 rounds (no trace continuation).
    
    This is a budget SCHEDULE, not adaptive allocation. Round 2 generates
    fresh samples; it does not continue traces from round 1 survivors.
    
    Primary metric: pass@k (any correct among ALL samples)
    Secondary metric: majority_correct (final majority vote)
    
    Args:
        problems: list of problem dicts with "id", "problem", "answer"
        model: model name
        temperature: sampling temperature
        max_tokens_cap: absolute cap on max_tokens per call (safety)
        budget_tokens: per-problem decoded token budget B
        N: initial samples in round 1 (default 16)
        K: survivors to round 2 (default 4)
        seed: base seed for deterministic sampling (derived per draw)
    
    Returns:
        list of result dicts with "id", "samples_taken", "decoded_tokens", "pass"
    """
    results = []

    for p in problems:
        problem_result = _two_phase_one_problem(
            problem=p,
            model=model,
            temperature=temperature,
            max_tokens_cap=max_tokens_cap,
            budget=budget_tokens,
            N=N,
            K=K,
            seed=seed,
        )
        results.append(problem_result)

    return results


# Backward compatibility aliases (emit warnings in main.py)
run_two_stage_self_consistency_budgeted = run_two_phase_budgeted_sampling
run_successive_halving_with_budget = run_two_phase_budgeted_sampling


def _two_phase_one_problem(
    *, problem, model, temperature, max_tokens_cap, budget, N, K, seed=None
):
    """Run two-phase budgeted sampling on a single problem.
    
    Budget accounting:
    - Round 1 gets 25% of budget B, split across N samples
    - Round 2 uses remaining budget (B - round1_actual) by generating batches
      of K samples until budget is exhausted (same scaling as uniform)
    - max_tokens per call uses max_tokens_cap (same as uniform) to allow
      comparable spending via multiple calls
      
    Primary metric: pass@k (any correct among ALL samples from both rounds)
    Secondary metric: majority_correct (final majority vote from round 2)
    """
    prompt = format_math_prompt(problem["problem"])
    gold_answer = problem["answer"]
    problem_id = str(problem["id"])

    # Round 1 budget: 25% of total, split across N samples
    round1_budget = int(0.25 * budget)
    round1_max_tokens = max(1, min(round1_budget // N, max_tokens_cap))

    # Round 1: generate N samples
    round1_samples = []
    spent = 0  # Track actual decoded tokens
    total_tokens_used = 0
    draw_idx = 0
    all_answers: list[str | None] = []
    total_calls = 0
    cap_hit_calls = 0

    for i in range(N):
        # Derive unique seed for this specific call
        call_seed = derive_call_seed(seed, problem_id, draw_idx)
        
        out = call_model(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=round1_max_tokens,
            seed=call_seed,
            cache_salt=(draw_idx if temperature != 0.0 else None),
        )
        total_calls += 1
        draw_idx += 1
        ctoks = out.completion_tokens or 0
        ptoks = out.prompt_tokens or 0
        ttoks = out.total_tokens
        if ttoks is None:
            ttoks = ptoks + ctoks
        if ctoks == round1_max_tokens:
            cap_hit_calls += 1

        spent += ctoks
        total_tokens_used += ttoks

        answer = extract_final_answer(out.text)
        all_answers.append(answer)
        round1_samples.append({
            "index": i,
            "text": out.text,
            "answer": answer,
            "tokens": ctoks,
        })

    # Score by answer frequency (self-consistency)
    answer_counts = Counter(
        s["answer"] for s in round1_samples if s["answer"] is not None
    )

    # Assign frequency score to each sample (0 if unparsable)
    for s in round1_samples:
        if s["answer"] is None:
            s["score"] = 0
        else:
            s["score"] = answer_counts[s["answer"]]

    # Select K survivors: sort by score desc, then by index asc (deterministic tie-break)
    sorted_samples = sorted(
        round1_samples, key=lambda s: (-s["score"], s["index"])
    )
    survivors = sorted_samples[:K]

    # Round 2: use remaining budget by generating batches of K samples
    # until budget is exhausted (matching uniform's scaling behavior)
    round2_answers = []
    round2_samples_taken = 0

    while spent < budget:
        # Check if we can afford another batch of K samples
        remaining = budget - spent
        if remaining < K:
            break
        
        # Use same max_tokens as uniform (max_tokens_cap) for comparable spending
        round2_max_tokens = max_tokens_cap
        
        # Generate K samples in this batch
        batch_answers = []
        batch_spent = 0
        batch_total_tokens = 0
        
        for _ in range(K):
            # Pre-check: stop if budget would be exceeded
            if spent + batch_spent >= budget:
                break
            
            # Derive unique seed for this specific call
            call_seed = derive_call_seed(seed, problem_id, draw_idx)
            
            out = call_model(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=round2_max_tokens,
                seed=call_seed,
                cache_salt=(draw_idx if temperature != 0.0 else None),
            )
            total_calls += 1
            draw_idx += 1
            ctoks = out.completion_tokens or 0
            ptoks = out.prompt_tokens or 0
            ttoks = out.total_tokens
            if ttoks is None:
                ttoks = ptoks + ctoks
            if ctoks == round2_max_tokens:
                cap_hit_calls += 1
            
            # Post-check: don't count if it would exceed budget
            if spent + batch_spent + ctoks > budget:
                break
            
            batch_spent += ctoks
            batch_total_tokens += ttoks
            answer = extract_final_answer(out.text)
            batch_answers.append(answer)
            all_answers.append(answer)
            round2_samples_taken += 1
        
        spent += batch_spent
        total_tokens_used += batch_total_tokens
        round2_answers.extend(batch_answers)
        
        # If we couldn't complete a full batch, stop
        if len(batch_answers) < K:
            break

    # If no round 2 samples generated, use round 1 survivors' answers
    if not round2_answers:
        round2_answers = [s["answer"] for s in survivors]

    # Final answer: majority vote across round 2 answers (for majority_correct)
    valid_round2 = [a for a in round2_answers if a is not None]
    if valid_round2:
        final_answer = Counter(valid_round2).most_common(1)[0][0]
    else:
        # Fallback to round 1 majority if round 2 all unparsable
        valid_r1 = [s["answer"] for s in survivors if s["answer"] is not None]
        if valid_r1:
            final_answer = Counter(valid_r1).most_common(1)[0][0]
        else:
            final_answer = None

    total_samples = N + round2_samples_taken
    
    # PRIMARY METRIC: pass@k = any correct among ALL samples (both rounds)
    pass_at_k = any(a == gold_answer for a in all_answers if a is not None)
    
    # SECONDARY METRIC: majority_correct = final answer matches gold
    majority_correct = (final_answer == gold_answer)

    # Assertion: spent should not exceed budget + small slack (allow 10 tokens slack)
    assert spent <= budget + 10, (
        f"Budget exceeded: spent {spent} > budget {budget} + 10 slack"
    )

    return {
        "id": problem["id"],
        "samples_taken": total_samples,
        "decoded_tokens": spent,
        "total_completion_tokens_used": spent,
        "total_tokens_used": total_tokens_used,
        "pass": pass_at_k,  # PRIMARY: pass@k (comparable with uniform)
        "pass_at_k": pass_at_k,
        "majority_correct": majority_correct,  # SECONDARY
        "total_calls": total_calls,
        "cap_hit_calls": cap_hit_calls,
    }


# Legacy aliases for backward compatibility
_two_stage_self_consistency_one_problem = _two_phase_one_problem
_successive_halving_one_problem = _two_phase_one_problem
