"""
Dataset utilities.

Expected JSONL format (one object per line):
  {"id": "...", "problem": "...", "answer": "..."}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl_dataset(path: str | Path) -> list[dict[str, str]]:
    """
    Load a JSONL dataset into a list of problem dicts.

    Returns items with keys: "id", "problem", "answer" (all strings).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    problems: list[dict[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj: dict[str, Any] = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {p}: {e}") from e

            # Canonical keys (as documented in README.md)
            pid = obj.get("id")
            problem = obj.get("problem")
            answer = obj.get("answer")

            # Best-effort back-compat for common alternates.
            if pid is None:
                pid = obj.get("qid") or obj.get("question_id") or str(line_no)
            if problem is None:
                problem = obj.get("question") or obj.get("prompt")
            if answer is None:
                answer = obj.get("final_answer") or obj.get("target")

            if problem is None or answer is None:
                raise ValueError(
                    f"Missing required keys on line {line_no} in {p}. "
                    f"Expected at least 'problem' and 'answer'. Got keys: {sorted(obj.keys())}"
                )

            problems.append(
                {
                    "id": str(pid),
                    "problem": str(problem),
                    "answer": str(answer).strip(),
                }
            )

    return problems

