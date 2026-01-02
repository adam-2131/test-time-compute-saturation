from utils import call_model
from parse import extract_final_answer
from prompting import format_math_prompt


def run_best_of_n_with_budget(problems, model, temperature, max_tokens, budget_tokens):
    results = []

    for p in problems:
        samples_taken = 0
        decoded_tokens = 0
        passed = False

        while True:
            if decoded_tokens >= budget_tokens:
                break

            out = call_model(
                format_math_prompt(p["problem"]),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            used = out.completion_tokens or 0
            if decoded_tokens + used > budget_tokens:
                break

            decoded_tokens += used
            samples_taken += 1

            pred = extract_final_answer(out.text)
            if pred == p["answer"]:
                passed = True

        results.append(
            {
                "id": p["id"],
                "samples_taken": samples_taken,
                "decoded_tokens": decoded_tokens,
                "pass": passed,
            }
        )

    return results

