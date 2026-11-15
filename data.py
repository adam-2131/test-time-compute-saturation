"""Loads a problem set from a .jsonl file (one object per line: id, problem, answer)."""

from __future__ import annotations

import json
from pathlib import Path


def load_jsonl_dataset(path):
    problems = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            # everything as strings, answers are compared as strings later
            try:
                problems.append({
                    "id": str(row["id"]),
                    "problem": str(row["problem"]),
                    "answer": str(row["answer"]).strip(),
                })
            except KeyError as err:
                raise ValueError(f"{path}, line {line_no}: missing key {err}") from err
    return problems
