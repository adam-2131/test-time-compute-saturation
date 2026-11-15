
from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from parse import extract_final_answer
from prompting import format_math_prompt
from utils import FatalModelError, call_model


PROBLEM_ATTEMPTS = 5


def _derive_seed(base_seed, *, problem_index, call_index, tag):
    data = f"{base_seed}:{problem_index}:{call_index}:{tag}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big") & 0x7FFFFFFF


def majority_stats(answers):
    valid_answers = [a for a in answers if a is not None]

    if not valid_answers:
        return None, 0.0, 0.0

    counts = Counter(valid_answers)
    total = len(valid_answers)

    max_count = max(counts.values())
    winners = [a for a, count in counts.items() if count == max_count]
    prediction = sorted(winners)[0]

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return prediction, max_count / total, entropy


def _vote(
    p_idx, problem, *, model, temperature, max_tokens,
    k_max, seed, min_samples, early_stop_stability,
    early_stop_confidence
):
    start = time.perf_counter()
    prompt = format_math_prompt(problem["problem"])
    gold = str(problem["answer"]).strip()

    answers = []
    k_used = 0
    previous_majority = None
    stable_for = 0

    for call_idx in range(k_max):
        result = call_model(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=_derive_seed(
                seed, problem_index=p_idx,
                call_index=call_idx, tag="fixed_k"
            ),
        )

        answers.append(extract_final_answer(result.text))
        k_used += 1

        if k_used < min_samples:
            continue

        majority, confidence, _ = majority_stats(answers)

        if confidence >= early_stop_confidence:
            break

        if majority is not None and majority == previous_majority:
            stable_for += 1
            if stable_for >= early_stop_stability:
                break
        else:
            stable_for = 1

        previous_majority = majority

    prediction, confidence, entropy = majority_stats(answers)

    return {
        "example_id": problem["id"],
        "k_max": k_max,
        "k_used": k_used,
        "prediction": prediction,
        "confidence": confidence,
        "entropy": entropy,
        "correct": prediction is not None and prediction == gold,
        "runtime_seconds": time.perf_counter() - start,
    }


def run_fixed_k_majority(
    problems, model, temperature, max_tokens, k_max, *,
    seed=0, early_stop_stability=50, early_stop_confidence=0.95,
    early_stop_min_samples=10, max_workers=1, progress=False
):
    min_samples = max(1, min(early_stop_min_samples, k_max))
    n = len(problems)

    def vote(p_idx, problem):
        last_error = None

        for attempt in range(PROBLEM_ATTEMPTS):
            try:
                return _vote(
                    p_idx,
                    problem,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    k_max=k_max,
                    seed=seed,
                    min_samples=min_samples,
                    early_stop_stability=early_stop_stability,
                    early_stop_confidence=early_stop_confidence,
                )
            except FatalModelError:
                raise
            except Exception as err:
                last_error = err
                time.sleep(min(60, 10 * (attempt + 1)))

        raise RuntimeError(
            f"problem {problem.get('id', p_idx)} failed after repeated retries: {last_error}"
        ) from last_error

    if max_workers <= 1:
        results = []

        for p_idx, problem in enumerate(problems):
            results.append(vote(p_idx, problem))

            if progress:
                print(f"  {len(results)}/{n}", flush=True)

        return results

    results = [None] * n

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(vote, i, p): i
            for i, p in enumerate(problems)
        }

        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()

            if progress:
                print(f"  {done}/{n}", flush=True)

    return results
