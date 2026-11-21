
from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from parse import extract_answer, normalize_gold
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


def is_truncated(output, max_tokens):
    # Older cache entries have no finish_reason; for them a response that used
    # the whole token budget counts as cut off.
    if output.finish_reason is not None:
        return output.finish_reason == "length"
    return output.completion_tokens is not None and output.completion_tokens >= max_tokens


def _vote(
    p_idx, problem, *, model, temperature, max_tokens,
    k_max, seed, min_samples, min_valid, early_stop_stability,
    early_stop_confidence
):
    start = time.perf_counter()
    prompt = format_math_prompt(problem["problem"])
    gold_answers = normalize_gold(problem["answer"])

    answers = []
    k_used = 0
    valid_votes = 0
    token_limit_votes = 0
    completion_tokens = 0
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

        truncated = is_truncated(result, max_tokens)
        answer = extract_answer(result.text, token_limit=truncated)

        answers.append(answer)
        k_used += 1
        valid_votes += answer is not None
        token_limit_votes += truncated
        completion_tokens += result.completion_tokens or 0

        # An invalid response is a drawn sample, but it neither supports nor
        # changes the majority, so it must not move confidence or stability.
        if k_used < min_samples or answer is None:
            continue

        majority, confidence, _ = majority_stats(answers)

        if majority == previous_majority:
            stable_for += 1
        else:
            stable_for = 1

        previous_majority = majority

        if valid_votes < min_valid:
            continue

        if confidence >= early_stop_confidence:
            break

        if stable_for >= early_stop_stability:
            break

    prediction, confidence, entropy = majority_stats(answers)

    return {
        "example_id": problem["id"],
        "k_max": k_max,
        "k_used": k_used,
        "valid_votes": valid_votes,
        "invalid_votes": k_used - valid_votes,
        "token_limit_votes": token_limit_votes,
        "prediction": prediction,
        "confidence": confidence,
        "entropy": entropy,
        "correct": prediction is not None and prediction in gold_answers,
        "completion_tokens": completion_tokens,
        "runtime_seconds": time.perf_counter() - start,
    }


def run_fixed_k_majority(
    problems,
    model,
    temperature,
    max_tokens,
    k_max,
    *,
    seed=0,
    early_stop_stability=50,
    early_stop_confidence=0.95,
    early_stop_min_samples=10,
    early_stop_min_valid=10,
    max_workers=1,
    progress=False,
):
    if k_max < 1:
        raise ValueError("k_max must be at least 1")

    if early_stop_stability < 1:
        raise ValueError("early_stop_stability must be at least 1")

    if early_stop_min_valid < 1:
        raise ValueError("early_stop_min_valid must be at least 1")

    if not 0.0 <= early_stop_confidence <= 1.0:
        raise ValueError("early_stop_confidence must be between 0 and 1")

    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

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
                    min_valid=early_stop_min_valid,
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
            pool.submit(vote, index, problem): index
            for index, problem in enumerate(problems)
        }

        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()

            if progress:
                print(f"  {done}/{n}", flush=True)

    return results
