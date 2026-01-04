from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import ParserError


@dataclass(frozen=True)
class RepoPaths:
    repo_root: Path
    results_dir: Path
    data_dir: Path


def get_repo_paths() -> RepoPaths:
    """
    Resolve repo-relative paths from the dashboard package location.
    This keeps the dashboard runnable from any working directory.
    """
    repo_root = Path(__file__).resolve().parents[1]
    return RepoPaths(
        repo_root=repo_root,
        results_dir=repo_root / "results",
        data_dir=repo_root / "data",
    )


_SEED_RE = re.compile(r"\bseed=(\d+)\b")
_CAP_HIT_RATE_RE = re.compile(r"\bcap_hit_rate=([0-9]+(?:\.[0-9]+)?)\b")


def parse_notes(notes: str) -> dict[str, int | float | None]:
    """
    Parse structured fields embedded in the freeform notes string.

    Expected patterns (as written by main.py):
      - seed=<int>
      - cap_hit_rate=<float>
    """
    notes = notes or ""
    seed: int | None = None
    cap_hit_rate: float | None = None

    m = _SEED_RE.search(notes)
    if m:
        try:
            seed = int(m.group(1))
        except ValueError:
            seed = None

    m = _CAP_HIT_RATE_RE.search(notes)
    if m:
        try:
            cap_hit_rate = float(m.group(1))
        except ValueError:
            cap_hit_rate = None

    return {"seed": seed, "cap_hit_rate": cap_hit_rate}


def discover_bootstrap_ci_files(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []
    files = sorted(results_dir.glob("bootstrap_ci_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def discover_extraction_audit_files(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []
    files = sorted(results_dir.glob("extraction_audit_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def load_scaling_results_csv(path: Path) -> pd.DataFrame:
    """
    Load results/scaling_results.csv with best-effort backwards compatibility.

    - Handles legacy rows missing pass@k/majority columns (results_logger.py can migrate,
      but we avoid relying on that for read-only usage).
    - Adds parsed columns: seed, cap_hit_rate (from notes).
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    # Read as strings first to avoid pandas NA coercions breaking robust parsing.
    # Some legacy raw logs contain occasional malformed rows (extra leading commas, etc.).
    # If pandas' fast parser fails, fall back to a tolerant CSV reader and normalize rows.
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except ParserError:
        df = _load_scaling_results_csv_fallback(path)

    # Ensure expected columns exist (schema v2).
    # Back-compat: if legacy "accuracy" exists, treat it as pass@k ONLY if pass_at_k_accuracy is missing/empty.
    expected_v2 = [
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
    for c in expected_v2:
        if c not in df.columns:
            df[c] = ""
    if "accuracy" in df.columns and ("pass_at_k_accuracy" not in df.columns or df["pass_at_k_accuracy"].astype(str).str.strip().eq("").all()):
        # Legacy fallback mapping: do NOT surface as "accuracy" in UI.
        df["pass_at_k_accuracy"] = df["accuracy"]

    # Parse numeric-ish columns.
    def _to_int(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").astype("Int64")

    def _to_float(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    df["compute_multiplier"] = _to_int(df["compute_multiplier"])
    df["pass_at_k_accuracy"] = _to_float(df["pass_at_k_accuracy"])
    df["majority_vote_accuracy"] = _to_float(df["majority_vote_accuracy"])
    df["total_decoded_tokens"] = _to_int(df["total_decoded_tokens"])
    df["num_problems"] = _to_int(df["num_problems"])

    # Parse notes fields.
    parsed = df["notes"].fillna("").map(parse_notes)
    df["seed"] = parsed.map(lambda d: d.get("seed"))
    df["cap_hit_rate"] = parsed.map(lambda d: d.get("cap_hit_rate"))

    return df


_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _load_scaling_results_csv_fallback(path: Path) -> pd.DataFrame:
    """
    Tolerant loader for legacy/malformed CSV rows.

    Supports common historical schemas:
      - v0 (10 cols): run_id,date,model,dataset,method,compute_multiplier,accuracy,total_decoded_tokens,num_problems,notes
      - v1 (12 cols): v0 + total_completion_tokens_used,total_tokens_used
      - current (14 cols): v1 + pass_at_k_accuracy,majority_vote_accuracy (inserted after accuracy)

    Also handles occasional extra leading empty column (e.g., rows starting with ',<run_id>,...').
    """
    # Fallback parser supports legacy schemas, but returns a schema-v2-ish frame
    # where the primary metric column is pass_at_k_accuracy.
    expected = [
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

    import csv

    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for fields in reader:
            if not fields:
                continue
            fields = [c.strip() for c in fields]
            if fields and fields[0]:
                fields[0] = fields[0].lstrip("\ufeff")

            # Skip headers (both canonical and legacy).
            if fields[0] in {"run_id", "date"}:
                continue
            if fields == expected:
                continue

            # Common malformed case: extra leading empty field, with run_id in second cell.
            if len(fields) >= 2 and fields[0] == "" and _RUN_ID_RE.match(fields[1]):
                fields = fields[1:]

            # Missing run_id variants: date appears first.
            if fields and _DATE_RE.match(fields[0]):
                fields = [""] + fields

            # Normalize schema lengths to legacy-14 first, then map to v2.
            # Legacy v0 (10): ... ,accuracy,total_decoded_tokens,num_problems,notes
            # Legacy v1 (12): ... ,accuracy,total_decoded_tokens,total_completion_tokens_used,total_tokens_used,num_problems,notes
            # Legacy v2 (14): ... ,accuracy,pass_at_k_accuracy,majority_vote_accuracy,total_decoded_tokens,...
            if len(fields) == 10:
                fields = fields[:7] + ["", ""] + [fields[7]] + ["", ""] + fields[8:]
            elif len(fields) == 12:
                fields = fields[:7] + ["", ""] + fields[7:]

            # Now map legacy-14 -> v2-13 by dropping "accuracy" and preferring pass_at_k_accuracy.
            if len(fields) == 14:
                legacy_accuracy = fields[6]
                legacy_passk = fields[7]
                legacy_majority = fields[8]
                final_passk = legacy_passk if legacy_passk else legacy_accuracy
                fields = fields[:6] + [final_passk, legacy_majority] + fields[9:]

            # Best-effort pad/truncate to v2 schema.
            if len(fields) < len(expected):
                fields = fields + [""] * (len(expected) - len(fields))
            if len(fields) > len(expected):
                fields = fields[: len(expected)]

            if len(fields) != len(expected):
                continue

            rows.append({k: v for k, v in zip(expected, fields)})

    return pd.DataFrame(rows, columns=expected)


def load_bootstrap_ci_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_extraction_audit_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    # Ensure key columns exist for older/hand-edited files.
    for c in [
        "problem_id",
        "gold_answer",
        "extracted_answer",
        "round",
        "draw_index",
        "completion_tokens",
        "cache_path",
        "raw_output",
    ]:
        if c not in df.columns:
            df[c] = ""
    return df


def safe_new_results_path(results_dir: Path, filename: str) -> Path:
    """
    Create a non-colliding output path under results/.
    If filename exists, suffix with _2, _3, ...
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    base = results_dir / filename
    if not base.exists():
        return base

    stem = base.stem
    suffix = base.suffix
    i = 2
    while True:
        candidate = results_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1




