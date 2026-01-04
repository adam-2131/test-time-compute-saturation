$ErrorActionPreference = "Stop"

# Windows PowerShell entrypoint for post-hoc analysis (no new model calls).
# Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\run_all_posthoc.ps1

$PY = $env:PYTHON
if (-not $PY) { $PY = "python" }

Write-Host "[posthoc] clean_results.py"
& $PY "scripts\clean_results.py"

Write-Host "[posthoc] rebuild_scaling_results.py"
& $PY "scripts\rebuild_scaling_results.py"

if (Test-Path "results\per_problem_results.jsonl") {
  Write-Host "[posthoc] bootstrap_ci.py (accuracy, diff=10,100)"
  & $PY "scripts\bootstrap_ci.py" "--metric" "accuracy" "--diff" "10,100" "--n-bootstrap" "10000" "--seed" "0"
} else {
  Write-Host "[posthoc] skipping bootstrap_ci.py (missing results\per_problem_results.jsonl)"
}

Write-Host "[posthoc] make_report_bundle.py"
& $PY "scripts\make_report_bundle.py"

Write-Host "[posthoc] done -> results\report_bundle\"


