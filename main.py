# main.py
from data import load_frontier_math
from eval import run_best_of_n_with_budget
from results_logger import append_result_row
from prompting import format_math_prompt
from utils import call_model

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7
MAX_TOKENS = 2048
DATASET_NAME = "toy"


def run():
    problems = load_frontier_math("data/frontier_math.jsonl")

    baseline_total_tokens = 0
    for p in problems:
        out = call_model(
            format_math_prompt(p["problem"]),
            model=MODEL,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
        baseline_total_tokens += out.completion_tokens or 0

    baseline_average_tokens = baseline_total_tokens / len(problems)

    multipliers = [1, 10, 100, 1000]
    rows = []

    for m in multipliers:
        # Budget is "decoded completion tokens" (ints); keep the budget integral.
        budget_tokens = int(m * baseline_average_tokens)
        results = run_best_of_n_with_budget(
            problems,
            MODEL,
            TEMPERATURE,
            MAX_TOKENS,
            budget_tokens,
        )
        accuracy = sum(1 for r in results if r["pass"]) / len(results)
        total_decoded_tokens_used = sum(r["decoded_tokens"] for r in results)
        rows.append((m, accuracy, total_decoded_tokens_used))
        append_result_row(
            model=MODEL,
            dataset=DATASET_NAME,
            method="uniform_best_of_n_budgeted",
            compute_multiplier=m,
            accuracy=accuracy,
            total_decoded_tokens=total_decoded_tokens_used,
            num_problems=len(problems),
            notes="sanity_check" if DATASET_NAME == "toy" else "",
        )

    header = f"{'multiplier':>10}  {'accuracy':>10}  {'total_decoded_tokens_used':>25}"
    print(header)
    print("-" * len(header))
    for m, acc, toks in rows:
        print(f"{m:>10}  {acc:>10.4f}  {toks:>25}")


if __name__ == "__main__":
    run()
