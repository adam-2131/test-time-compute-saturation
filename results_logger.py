# =============================================================================
# LEGACY: Token-budgeted experiment logging
# =============================================================================
# This module supports the legacy token-budgeted experiments (uniform_best_of_n_budgeted,
# two_phase_budgeted_sampling). The primary experiment path is now fixed-k majority
# voting, which writes directly to experiments/k_*.csv.
#
# Keep for backwards compatibility with existing results/scaling_results.csv artifacts.
# =============================================================================

from __future__ import annotations

import csv
import os
import re
from datetime import date, datetime
from typing import Any

# Canonical (current) schema used by main.py and the dashboard.
HEADER: list[str] = [
    "run_id",
    "date",
    "model",
    "dataset",
    "method",
    "compute_multiplier",
    "pass_at_k_accuracy",
    "majority_vote_accuracy",
    "total_decoded_tokens",
    "total_completion_tokens_used",
    "total_tokens_used",
    "num_problems",
    "notes",
]

_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}$")


def _generate_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    run_id = now.strftime("%Y%m%d-%H%M%S")
    if not run_id or not _RUN_ID_RE.fullmatch(run_id):
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not run_id:
        raise RuntimeError("Failed to generate non-empty run_id")
    return run_id


_RUN_ID = _generate_run_id()


def get_run_id() -> str:
    """Stable run id for the current Python process."""
    return _RUN_ID


def _looks_like_header(row: list[str]) -> bool:
    row = [c.strip() for c in row]
    if row and row[0]:
        row[0] = row[0].lstrip("\ufeff")
    return bool(row) and all(c in set(HEADER) or c == "accuracy" for c in row)


def _normalize_legacy_row(fields: list[str]) -> dict[str, str] | None:
    """
    Best-effort normalization across historical schemas:
      - v0 (10 cols): run_id,date,model,dataset,method,compute_multiplier,accuracy,total_decoded_tokens,num_problems,notes
      - v1 (12 cols): v0 + total_completion_tokens_used,total_tokens_used
      - v2 (14 cols): v1 + pass_at_k_accuracy,majority_vote_accuracy (inserted after accuracy)
      - current (13 cols): HEADER (accuracy removed; pass_at_k_accuracy primary)
    """
    fields = [c.strip() for c in fields]
    if fields and fields[0]:
        fields[0] = fields[0].lstrip("\ufeff")

    # Common malformed case: extra leading empty column.
    if len(fields) >= 2 and fields[0] == "" and _RUN_ID_RE.fullmatch(fields[1]):
        fields = fields[1:]

    # If already current schema.
    if len(fields) == len(HEADER):
        row = dict(zip(HEADER, fields))
        if not row.get("run_id") or not _RUN_ID_RE.fullmatch(row["run_id"]):
            row["run_id"] = _generate_run_id()
        return row

    # v0
    if len(fields) == 10:
        # Insert pass_at_k_accuracy, majority_vote_accuracy after accuracy; then token columns.
        # v0 -> legacy-14 (with blanks for new cols)
        fields = fields[:7] + ["", ""] + [fields[7]] + ["", ""] + fields[8:]

    # v1
    if len(fields) == 12:
        # v1 -> legacy-14 by inserting pass_at_k_accuracy, majority_vote_accuracy after accuracy.
        fields = fields[:7] + ["", ""] + fields[7:]

    # v2 legacy-14: drop "accuracy" and prefer pass_at_k_accuracy if present.
    if len(fields) == 14:
        legacy_accuracy = fields[6]
        legacy_passk = fields[7]
        legacy_majority = fields[8]
        final_passk = legacy_passk if legacy_passk else legacy_accuracy
        # Map to current HEADER (13 cols): remove accuracy cell.
        mapped = fields[:6] + [final_passk, legacy_majority] + fields[9:]
        if len(mapped) != len(HEADER):
            return None
        row = dict(zip(HEADER, mapped))
        if not row.get("run_id") or not _RUN_ID_RE.fullmatch(row["run_id"]):
            row["run_id"] = _generate_run_id()
        return row

    return None


def _ensure_results_csv_schema(csv_path: str) -> None:
    """
    Ensure results/scaling_results.csv exists and matches HEADER.
    If it exists with a legacy schema, rewrite it into canonical form.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=HEADER).writeheader()
        return

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as rf:
        rows = list(csv.reader(rf))

    if not rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=HEADER).writeheader()
        return

    first = [c.strip() for c in rows[0]]
    if first and first[0]:
        first[0] = first[0].lstrip("\ufeff")

    header_ok = first == HEADER
    data_rows = rows[1:] if header_ok else rows

    # If header is correct, and all rows are well-formed, do nothing.
    if header_ok:
        for r in data_rows:
            if not r:
                continue
            if _looks_like_header(r):
                # Stray header rows; rewrite to drop them.
                header_ok = False
                break
            if len(r) != len(HEADER):
                header_ok = False
                break
        if header_ok:
            return

    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as wf:
        writer = csv.DictWriter(wf, fieldnames=HEADER)
        writer.writeheader()
        for r in data_rows:
            if not r:
                continue
            if _looks_like_header(r):
                continue
            row = _normalize_legacy_row(r)
            if row is None:
                continue
            writer.writerow(row)
    os.replace(tmp_path, csv_path)


def append_result_row(
    *,
    model: str,
    dataset: str,
    method: str,
    compute_multiplier: int,
    pass_at_k_accuracy: float,
    majority_vote_accuracy: float,
    total_decoded_tokens: int,
    total_completion_tokens_used: int,
    total_tokens_used: int,
    num_problems: int,
    notes: str = "",
    run_id: str | None = None,
) -> None:
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "scaling_results.csv")
    _ensure_results_csv_schema(csv_path)

    rid = run_id or get_run_id()
    if not rid or not _RUN_ID_RE.fullmatch(rid):
        rid = _generate_run_id()

    row: dict[str, Any] = {
        "run_id": rid,
        "date": date.today().isoformat(),
        "model": model,
        "dataset": dataset,
        "method": method,
        "compute_multiplier": int(compute_multiplier),
        "pass_at_k_accuracy": float(pass_at_k_accuracy),
        "majority_vote_accuracy": float(majority_vote_accuracy),
        "total_decoded_tokens": int(total_decoded_tokens),
        "total_completion_tokens_used": int(total_completion_tokens_used),
        "total_tokens_used": int(total_tokens_used),
        "num_problems": int(num_problems),
        "notes": notes or "",
    }
    row = {k: row.get(k, "") for k in HEADER}

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=HEADER).writerow(row)

