"""
Build `data/aime.jsonl` from the Hugging Face dataset `gneubig/aime-1983-2024`.

Each output line is a JSON object:
  {"id": <string>, "problem": <string>, "answer": <string>}

Run:
  python scripts/build_aime_jsonl.py
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from datasets import load_dataset


def _first_present(row: Dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _stable_id(row: Dict[str, Any], idx: int) -> str:
    # Prefer explicit dataset ID if present.
    explicit = _first_present(row, ["ID", "id"])
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()

    # Otherwise synthesize from year / problem number / part if present.
    year = _first_present(row, ["Year", "year"])
    prob = _first_present(row, ["Problem Number", "problem_number", "problem"])
    part = _first_present(row, ["Part", "part"])

    pieces: list[str] = []
    if year is not None:
        pieces.append(str(year).strip())
    if prob is not None:
        pieces.append(str(prob).strip())
    if part is not None and str(part).strip():
        pieces.append(str(part).strip())

    if pieces:
        return "-".join(pieces)
    return str(idx)


def main() -> None:
    ds = load_dataset("gneubig/aime-1983-2024")
    split = ds["train"] if isinstance(ds, dict) and "train" in ds else next(iter(ds.values()))

    out_path = os.path.join("data", "aime.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(split):
            problem = _first_present(row, ["Question", "question", "Problem", "problem"])
            answer = _first_present(row, ["Answer", "answer"])
            rec = {
                "id": _stable_id(row, idx),
                "problem": "" if problem is None else str(problem),
                "answer": "" if answer is None else str(answer),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"wrote {n} rows to {out_path}")


if __name__ == "__main__":
    main()




