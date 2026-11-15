
import json
from pathlib import Path

from datasets import load_dataset


OUT_PATH = Path("data/gsm8k.jsonl")


def gold_answer(solution):
    return solution.split("####")[-1].replace(",", "").strip()


def main():
    rows = load_dataset("gsm8k", "main", split="test")

    OUT_PATH.parent.mkdir(exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for idx, row in enumerate(rows):
            record = {
                "id": f"gsm8k-test-{idx}",
                "problem": row["question"].strip(),
                "answer": gold_answer(row["answer"]),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
