#!/usr/bin/env python3
"""
Create a lightweight snapshot of results from a run.

Writes:
  - results/snapshots/<run_id>_scaling_results_snapshot.csv
  - results/snapshots/<run_id>_manifest.json (sha256 hashes)

Policy:
  - Requires results/scaling_results.csv to exist locally (fails with helpful message if missing)
  - Excludes toy/sanity_check datasets (only real datasets: aime, frontier_math)
  - Snapshot folder contains ONLY:
      - <run_id>_scaling_results_snapshot.csv
      - <run_id>_manifest.json
  - Manifest always includes sha256+bytes for snapshot CSV
  - Manifest optionally includes sha256+bytes for local:
      - results/per_problem_results.jsonl (hashed only; not copied)
      - results/bootstrap_ci_<run_id>_<metric>.json (hashed only; not copied)

Usage:
  python scripts/make_snapshot.py
  python scripts/make_snapshot.py --run-id 20260104-123456
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


# Canonical schema (must match results_logger.py)
CANONICAL_COLUMNS = [
    "run_id", "date", "model", "dataset", "method", "compute_multiplier",
    "pass_at_k_accuracy", "majority_vote_accuracy",
    "total_decoded_tokens", "total_completion_tokens_used", "total_tokens_used",
    "num_problems", "notes"
]

# Valid datasets for snapshots (exclude toy/sanity_check)
VALID_DATASETS = {"aime", "frontier_math"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_scaling_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _latest_run_id(rows: list[dict[str, str]]) -> str | None:
    run_ids = [r.get("run_id", "") for r in rows if r.get("run_id")]
    if not run_ids:
        return None
    # run_id format is YYYYMMDD-HHMMSS; lexical sort matches time ordering.
    return sorted(run_ids)[-1]


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    """Normalize row to canonical schema, handling legacy 'accuracy' column."""
    out = {}
    for col in CANONICAL_COLUMNS:
        if col in row:
            out[col] = row[col]
        elif col == "pass_at_k_accuracy" and "accuracy" in row:
            # Legacy migration: accuracy -> pass_at_k_accuracy
            out[col] = row.get("accuracy", "")
        else:
            out[col] = ""
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a snapshot of results from a run.")
    parser.add_argument("--run-id", default=None, help="Run ID to snapshot (default: latest)")
    parser.add_argument("--scaling-csv", default="results/scaling_results.csv", help="Source CSV")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    snapshots_dir = results_dir / "snapshots"

    # Require canonical local CSV (do not fall back to legacy raw/clean logs).
    scaling_csv = repo_root / args.scaling_csv
    if not scaling_csv.exists():
        print(f"ERROR: required local results file not found: {args.scaling_csv}")
        print()
        print("This repo does NOT commit results/scaling_results.csv. It is generated locally by running:")
        print("  python main.py --dataset-path data/aime.jsonl --dataset-name aime --limit 50 --multipliers 1,10,100")
        print()
        print("Then create a snapshot:")
        print("  python scripts/make_snapshot.py")
        return

    all_rows = _load_scaling_results(scaling_csv)
    if not all_rows:
        print("ERROR: no data rows in scaling results CSV")
        print("Run experiments first: python main.py")
        return

    run_id = args.run_id or _latest_run_id(all_rows)
    if not run_id:
        print("ERROR: could not determine run_id")
        return

    # Filter to this run_id
    rows = [r for r in all_rows if r.get("run_id") == run_id]
    if not rows:
        print(f"ERROR: no rows found for run_id={run_id}")
        available = sorted(set(r.get("run_id", "") for r in all_rows if r.get("run_id")))
        if available:
            print(f"Available run_ids: {', '.join(available[-5:])}")
        return

    # Filter to real datasets only (exclude toy/sanity_check)
    real_rows = [r for r in rows if r.get("dataset") in VALID_DATASETS]
    if not real_rows:
        print(f"WARNING: no real dataset rows (aime/frontier_math) found for run_id={run_id}")
        print("Only toy/sanity_check data present. Creating snapshot anyway for debugging.")
        real_rows = rows  # Allow for debugging, but warn

    # Create snapshots directory
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Write snapshot CSV with canonical schema
    snapshot_csv = snapshots_dir / f"{run_id}_scaling_results_snapshot.csv"
    with open(snapshot_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        for row in real_rows:
            writer.writerow(_normalize_row(row))

    print(f"wrote {snapshot_csv}")

    # Build manifest
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "source_csv": str(scaling_csv.relative_to(repo_root)),
        "num_rows": len(real_rows),
        "datasets": sorted(set(r.get("dataset", "") for r in real_rows if r.get("dataset"))),
        "methods": sorted(set(r.get("method", "") for r in real_rows if r.get("method"))),
        "outputs": {},
        "inputs": {},
    }

    # Hash snapshot CSV
    manifest["outputs"][snapshot_csv.name] = {
        "sha256": _sha256_file(snapshot_csv),
        "bytes": snapshot_csv.stat().st_size,
        "rows": len(real_rows),
    }

    # Hash bootstrap CI if present (do not copy into snapshots).
    for metric in ["accuracy", "pass_at_k", "majority_correct"]:
        ci_file = results_dir / f"bootstrap_ci_{run_id}_{metric}.json"
        if ci_file.exists():
            manifest["inputs"][ci_file.name] = {
                "sha256": _sha256_file(ci_file),
                "bytes": ci_file.stat().st_size,
            }
            print(f"found {ci_file.name} (hashed into manifest)")

    # Hash per_problem_results.jsonl if present (do not copy into snapshots).
    per_problem = results_dir / "per_problem_results.jsonl"
    if per_problem.exists():
        manifest["inputs"][per_problem.name] = {
            "sha256": _sha256_file(per_problem),
            "bytes": per_problem.stat().st_size,
        }
        print(f"found {per_problem.name} (hashed into manifest)")

    # Write manifest
    manifest_path = snapshots_dir / f"{run_id}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {manifest_path}")
    print()
    print(f"Snapshot complete for run_id={run_id}")
    print(f"  Datasets: {', '.join(manifest['datasets'])}")
    print(f"  Rows: {len(real_rows)}")


if __name__ == "__main__":
    main()
