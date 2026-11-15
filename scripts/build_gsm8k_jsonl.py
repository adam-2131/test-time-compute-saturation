"""Build data/gsm8k.jsonl from the Hugging Face dataset gsm8k (config "main",
test split). The answer is the number after "####" in the solution text.
"""

from __future__ import annotations

import json
import os
import re

from datasets import load_dataset

_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$")


def _extract_gold(answer_field):
    m = _ANSWER_RE.search(answer_field.strip())
    raw = m.group(1) if m else answer_field.strip()
    return raw.replace(",", "").strip()


def main():
    ds = load_dataset("gsm8k", "main")
    split = ds["test"] if "test" in ds else next(iter(ds.values()))

    out_path = os.path.join("data", "gsm8k.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(split):
            problem = str(row.get("question", "")).strip()
            answer = _extract_gold(str(row.get("answer", "")))
            rec = {
                "id": f"gsm8k-test-{idx}",
                "problem": problem,
                "answer": answer,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"wrote {n} rows to {out_path}")


if __name__ == "__main__":
    main()
