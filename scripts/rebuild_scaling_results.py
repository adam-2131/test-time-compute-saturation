#!/usr/bin/env python3
"""
LEGACY: Rebuild results/scaling_results.csv from results/scaling_results_raw.csv.

This script supports the legacy token-budgeted experiments. The primary experiment
path is now fixed-k majority voting (--method fixed_k_majority).

Includes ONLY rows that satisfy:
- run_id is non-empty OR can be backfilled as legacy_<line_number>
- dataset in {"aime", "frontier_math"}
- num_problems >= 10
- notes contains "subset="

Preserves original values otherwise.
"""
import csv
import os
import re

HEADER = [
    "run_id",
    "date",
    "model",
    "dataset",
    "method",
    "compute_multiplier",
    "accuracy",
    "pass_at_k_accuracy",
    "majority_vote_accuracy",
    "total_decoded_tokens",
    "total_completion_tokens_used",
    "total_tokens_used",
    "num_problems",
    "notes",
]

RAW_PATH = os.path.join("results", "scaling_results_raw.csv")
OUT_PATH = os.path.join("results", "scaling_results.csv")

RUN_ID_RE = re.compile(r"^\d{8}-\d{6}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_row(fields: list[str], line_number: int) -> dict[str, str] | None:
    """
    Parse a row from the raw CSV.
    Returns a dict with all HEADER fields, or None if the row is malformed.
    """
    # Strip whitespace and BOM
    fields = [f.strip() for f in fields]
    if fields:
        fields[0] = fields[0].lstrip("\ufeff")

    # Handle various malformed row formats
    # Case 1: Empty leading field with correct run_id in second position
    if len(fields) in {11, 13, 15} and fields[0] == "" and RUN_ID_RE.match(fields[1]):
        fields = fields[1:]
    # Case 2: Missing run_id (9 fields, date in first position)
    elif len(fields) in {9, 11, 13} and DATE_RE.match(fields[0]):
        fields = [""] + fields
    # Case 3: Extra empty leading field, no run_id
    elif len(fields) in {10, 12, 14} and fields[0] == "" and DATE_RE.match(fields[1]):
        pass  # run_id will be backfilled
    # Case 4: Normal row (legacy 10-field, old 12-field, or new 14-field)
    elif len(fields) in {10, 12, 14}:
        pass
    else:
        return None

    # Legacy rows omit the new accuracy + token columns; pad them.
    if len(fields) == 10:
        fields = fields[:7] + ["", ""] + [fields[7]] + ["", ""] + fields[8:]
    # Older rows omit the new accuracy columns; pad them.
    if len(fields) == 12:
        fields = fields[:7] + ["", ""] + fields[7:]
    if len(fields) != len(HEADER):
        return None

    row = dict(zip(HEADER, fields))

    # Backfill empty run_id
    if not row["run_id"] or not RUN_ID_RE.match(row["run_id"]):
        row["run_id"] = f"legacy_{line_number}"

    return row


def _is_valid_row(row: dict[str, str]) -> bool:
    """Check if a row meets inclusion criteria."""
    # Must have non-empty run_id (should always be true after backfill)
    if not row.get("run_id"):
        return False

    # dataset must be aime or frontier_math
    if row.get("dataset") not in {"aime", "frontier_math"}:
        return False

    # num_problems must be >= 10
    try:
        num_problems = int(row.get("num_problems", "0"))
        if num_problems < 10:
            return False
    except ValueError:
        return False

    # notes must contain "subset="
    notes = row.get("notes", "")
    if "subset=" not in notes:
        return False

    return True


def main() -> None:
    if not os.path.exists(RAW_PATH):
        print(f"error: {RAW_PATH} not found")
        return

    os.makedirs("results", exist_ok=True)

    valid_rows: list[dict[str, str]] = []

    with open(RAW_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for line_number, fields in enumerate(reader, start=1):
            # Skip header rows
            if fields and fields[0] in {"run_id", "date"}:
                continue
            if fields == HEADER:
                continue

            row = _parse_row(fields, line_number)
            if row is None:
                continue

            if _is_valid_row(row):
                valid_rows.append(row)

    # Write clean output
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for row in valid_rows:
            writer.writerow(row)

    print(f"rebuilt scaling_results.csv with {len(valid_rows)} rows")


if __name__ == "__main__":
    main()

