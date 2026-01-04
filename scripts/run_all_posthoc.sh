#!/usr/bin/env bash
set -euo pipefail

# Cross-platform-ish shell entrypoint for post-hoc analysis (no new model calls).
# Intended for macOS/Linux and Git-Bash environments on Windows.

PY="${PYTHON:-python}"

echo "[posthoc] clean_results.py"
$PY scripts/clean_results.py

echo "[posthoc] rebuild_scaling_results.py"
$PY scripts/rebuild_scaling_results.py

if [[ -f "results/per_problem_results.jsonl" ]]; then
  echo "[posthoc] bootstrap_ci.py (accuracy, diff=10,100)"
  $PY scripts/bootstrap_ci.py --metric accuracy --diff 10,100 --n-bootstrap 10000 --seed 0
else
  echo "[posthoc] skipping bootstrap_ci.py (missing results/per_problem_results.jsonl)"
fi

echo "[posthoc] make_report_bundle.py"
$PY scripts/make_report_bundle.py

echo "[posthoc] done -> results/report_bundle/"


