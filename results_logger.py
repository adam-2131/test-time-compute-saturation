import csv
import os
import re
from datetime import date, datetime
from typing import Dict, List, Optional

HEADER_STR = (
    "run_id,date,model,dataset,method,compute_multiplier,accuracy,"
    "pass_at_k_accuracy,majority_vote_accuracy,"
    "total_decoded_tokens,total_completion_tokens_used,total_tokens_used,"
    "num_problems,notes"
)
HEADER = HEADER_STR.split(",")


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
    return all(cell in set(HEADER) for cell in row)


def _normalize_row_to_header(row: List[str]) -> Dict[str, str]:
    """
    Best-effort migration for existing rows so the file stays schema-correct.
    Produces a dict with *all* keys in HEADER.
    """
    # Normalize whitespace and any stray BOM markers.
    row = [cell.strip() for cell in row]
    if row:
        row[0] = row[0].lstrip("\ufeff")

    # Handle known legacy schemas safely (we've inserted new columns in the middle over time).
    #
    # Legacy v0 (10 cols):
    #   run_id,date,model,dataset,method,compute_multiplier,accuracy,total_decoded_tokens,num_problems,notes
    # Legacy v1 (12 cols):
    #   run_id,date,model,dataset,method,compute_multiplier,accuracy,total_decoded_tokens,total_completion_tokens_used,total_tokens_used,num_problems,notes
    # Current (14 cols):
    #   run_id,date,model,dataset,method,compute_multiplier,accuracy,pass_at_k_accuracy,majority_vote_accuracy,
    #   total_decoded_tokens,total_completion_tokens_used,total_tokens_used,num_problems,notes

    # Missing run_id variants (pad on the left first, based on legacy lengths).
    if len(row) in {9, 11, 13}:
        row = [""] + row

    # Expand legacy v0 (10 -> 14) by inserting blanks in the right places.
    if len(row) == 10:
        # Insert pass@k + majority after accuracy; and token totals after total_decoded_tokens.
        row = row[:7] + ["", ""] + [row[7]] + ["", ""] + row[8:]

    # Expand legacy v1 (12 -> 14) by inserting blanks after accuracy.
    if len(row) == 12:
        row = row[:7] + ["", ""] + row[7:]

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
            if cleaned == HEADER:
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
    accuracy: float,
    total_decoded_tokens: int,
    num_problems: int,
    pass_at_k_accuracy: float | None = None,
    majority_vote_accuracy: float | None = None,
    total_completion_tokens_used: int | None = None,
    total_tokens_used: int | None = None,
    notes: str = "",
    run_id: Optional[str] = None,
) -> None:
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
        "accuracy": accuracy,
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

