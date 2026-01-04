"""
Results logging utilities for test-time compute experiments.

Schema (v2 - reviewer-safe):
  - Removed redundant "accuracy" column (was duplicate of pass_at_k_accuracy)
  - Primary metric: pass_at_k_accuracy
  - Secondary metric: majority_vote_accuracy
"""
import csv
import os
import re
from datetime import date, datetime
from typing import Dict, List, Optional

# Canonical schema (v2): no redundant "accuracy" column
HEADER_STR = (
    "run_id,date,model,dataset,method,compute_multiplier,"
    "pass_at_k_accuracy,majority_vote_accuracy,"
    "total_decoded_tokens,total_completion_tokens_used,total_tokens_used,"
    "num_problems,notes"
)
HEADER = HEADER_STR.split(",")

# Legacy schema columns for migration
LEGACY_ACCURACY_HEADER = [
    "run_id", "date", "model", "dataset", "method", "compute_multiplier",
    "accuracy", "pass_at_k_accuracy", "majority_vote_accuracy",
    "total_decoded_tokens", "total_completion_tokens_used", "total_tokens_used",
    "num_problems", "notes"
]


def _generate_run_id(now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    run_id = now.strftime("%Y%m%d-%H%M%S")
    # Defensive: ensure non-empty and correct shape.
    if not run_id or not re.fullmatch(r"\d{8}-\d{6}", run_id):
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Final guard: never allow empty run_id.
    if not run_id:
        raise RuntimeError("Failed to generate non-empty run_id")
    return run_id


_RUN_ID = _generate_run_id()


def get_run_id() -> str:
    """
    Return the run_id for the current process/session.
    This is used to ensure a single sweep (multiple multipliers) shares one run_id.
    """
    return _RUN_ID


def _looks_like_header(row: List[str]) -> bool:
    if not row:
        return False
    # If every cell is one of our known column names, treat as a header row.
    all_known = set(HEADER) | set(LEGACY_ACCURACY_HEADER)
    return all(cell in all_known for cell in row)


def _normalize_row_to_header(row: List[str]) -> Dict[str, str]:
    """
    Best-effort migration for existing rows so the file stays schema-correct.
    Produces a dict with *all* keys in HEADER.
    """
    # Normalize whitespace and any stray BOM markers.
    row = [cell.strip() for cell in row]
    if row:
        row[0] = row[0].lstrip("\ufeff")

    # Handle legacy schemas:
    # Legacy v0 (10 cols): run_id,date,model,dataset,method,compute_multiplier,accuracy,total_decoded_tokens,num_problems,notes
    # Legacy v1 (12 cols): + total_completion_tokens_used,total_tokens_used
    # Legacy v2 (14 cols): + pass_at_k_accuracy,majority_vote_accuracy (redundant accuracy)
    # Current (13 cols): no "accuracy" column
    
    # Missing run_id variants (pad on the left first).
    if len(row) in {9, 11, 12}:
        row = [""] + row

    # Expand legacy v0 (10 -> 13)
    if len(row) == 10:
        # run_id,date,model,dataset,method,compute_multiplier,accuracy,total_decoded_tokens,num_problems,notes
        # -> insert pass@k=accuracy, majority="", then token totals
        accuracy_val = row[6]
        row = row[:6] + [accuracy_val, ""] + [row[7]] + ["", ""] + row[8:]
    
    # Expand legacy v1 (12 -> 13)
    if len(row) == 12:
        # run_id,date,model,dataset,method,compute_multiplier,accuracy,total_decoded_tokens,total_completion_tokens_used,total_tokens_used,num_problems,notes
        accuracy_val = row[6]
        row = row[:6] + [accuracy_val, ""] + row[7:]
    
    # Handle legacy v2 with redundant accuracy column (14 -> 13)
    if len(row) == 14:
        # Old: run_id,date,model,dataset,method,compute_multiplier,accuracy,pass_at_k_accuracy,majority_vote_accuracy,...
        # New: run_id,date,model,dataset,method,compute_multiplier,pass_at_k_accuracy,majority_vote_accuracy,...
        # Use pass_at_k_accuracy if present, else fall back to accuracy
        accuracy_val = row[6]  # old "accuracy"
        pass_at_k_val = row[7]  # pass_at_k_accuracy
        majority_val = row[8]   # majority_vote_accuracy
        # Prefer pass_at_k_accuracy if non-empty, else use accuracy
        final_pass_at_k = pass_at_k_val if pass_at_k_val else accuracy_val
        row = row[:6] + [final_pass_at_k, majority_val] + row[9:]

    # Final normalization: pad/truncate to match current schema.
    if len(row) < len(HEADER):
        row = row + [""] * (len(HEADER) - len(row))
    if len(row) > len(HEADER):
        row = row[: len(HEADER)]

    out = {k: v for k, v in zip(HEADER, row)}
    # Ensure run_id is present and valid even for migrated rows.
    if (not out["run_id"]) or (not re.fullmatch(r"\d{8}-\d{6}", out["run_id"])):
        out["run_id"] = _generate_run_id()
    return out


def _ensure_results_csv_schema(csv_path: str) -> None:
    """
    Ensures the CSV exists and its first row matches HEADER exactly.
    If the file exists with a different header, it is rewritten into a
    schema-correct form (best-effort row normalization).
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)
            writer.writeheader()
        return

    # utf-8-sig strips a UTF-8 BOM if present.
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as rf:
        rows = list(csv.reader(rf))

    if not rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)
            writer.writeheader()
        return

    first = [cell.strip() for cell in rows[0]]
    if first:
        first[0] = first[0].lstrip("\ufeff")

    header_ok = first == HEADER
    data_rows = rows[1:] if header_ok else rows

    needs_rewrite = not header_ok
    if not needs_rewrite:
        for r in data_rows:
            if not r:
                continue
            cleaned = [cell.strip() for cell in r]
            if cleaned:
                cleaned[0] = cleaned[0].lstrip("\ufeff")
            if _looks_like_header(cleaned) or cleaned == HEADER:
                needs_rewrite = True
                break
            if len(cleaned) != len(HEADER):
                needs_rewrite = True
                break
            run_id_cell = cleaned[0] if cleaned else ""
            if run_id_cell and not re.fullmatch(r"\d{8}-\d{6}", run_id_cell):
                needs_rewrite = True
                break

    if not needs_rewrite:
        return

    # Rewrite/migrate to match our canonical header.
    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as wf:
        writer = csv.DictWriter(wf, fieldnames=HEADER)
        writer.writeheader()
        for row in data_rows:
            if not row:
                continue
            cleaned = [cell.strip() for cell in row]
            if cleaned:
                cleaned[0] = cleaned[0].lstrip("\ufeff")
            if cleaned == HEADER or cleaned == LEGACY_ACCURACY_HEADER:
                continue
            if _looks_like_header(cleaned):
                continue
            writer.writerow(_normalize_row_to_header(cleaned))
    os.replace(tmp_path, csv_path)


def append_result_row(
    model: str,
    dataset: str,
    method: str,
    compute_multiplier: int,
    pass_at_k_accuracy: float,
    total_decoded_tokens: int,
    num_problems: int,
    majority_vote_accuracy: float | None = None,
    total_completion_tokens_used: int | None = None,
    total_tokens_used: int | None = None,
    notes: str = "",
    run_id: Optional[str] = None,
) -> None:
    """
    Append a result row to results/scaling_results.csv.
    
    Args:
        model: Model name (e.g., "gpt-4o-mini")
        dataset: Dataset name (e.g., "aime", "frontier_math")
        method: Canonical method name (e.g., "uniform_best_of_n_budgeted", "two_phase_budgeted_sampling")
        compute_multiplier: Budget multiplier (1, 10, 100, etc.)
        pass_at_k_accuracy: PRIMARY METRIC - fraction of problems with at least one correct sample
        total_decoded_tokens: Sum of decoded tokens across all problems
        num_problems: Number of problems evaluated
        majority_vote_accuracy: SECONDARY METRIC - fraction where majority vote is correct
        total_completion_tokens_used: Same as total_decoded_tokens (for accounting)
        total_tokens_used: Total tokens including prompt (for cost tracking)
        notes: Free-form notes (subset=, seed=, cap_hit_rate=, etc.)
        run_id: Unique run identifier (auto-generated if not provided)
    """
    os.makedirs("results", exist_ok=True)

    csv_path = os.path.join("results", "scaling_results.csv")
    _ensure_results_csv_schema(csv_path)

    run_id = run_id or (_RUN_ID if isinstance(_RUN_ID, str) and _RUN_ID else _generate_run_id())
    if not re.fullmatch(r"\d{8}-\d{6}", run_id):
        run_id = _generate_run_id()
    if not run_id:
        raise RuntimeError("Refusing to write results row with empty run_id")

    row: Dict[str, object] = {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "model": model,
        "dataset": dataset,
        "method": method,
        "compute_multiplier": compute_multiplier,
        "pass_at_k_accuracy": pass_at_k_accuracy,
        "majority_vote_accuracy": majority_vote_accuracy,
        "total_decoded_tokens": total_decoded_tokens,
        "total_completion_tokens_used": total_completion_tokens_used,
        "total_tokens_used": total_tokens_used,
        "num_problems": num_problems,
        "notes": notes,
    }

    # Defensive: ensure all required keys exist (and no unexpected schema changes slip in).
    # Also normalize None -> "" so downstream scripts can keep parsing ints safely.
    row = {k: ("" if row.get(k, "") is None else row.get(k, "")) for k in HEADER}
    if not row.get("run_id"):
        raise RuntimeError("Refusing to write results row with empty run_id")

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writerow(row)
