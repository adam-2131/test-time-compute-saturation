
import json
from pathlib import Path

from datasets import load_dataset


OUT_PATH = Path("data/aime.jsonl")


def main():
    rows = load_dataset("gneubig/aime-1983-2024", split="train")

    OUT_PATH.parent.mkdir(exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for row in rows:
            record = {
                "id": row["ID"],
                "problem": row["Question"],
                "answer": row["Answer"],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
