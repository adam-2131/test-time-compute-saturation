## Results & Audit Dashboard (Streamlit, read-only)

This is a tiny inspector UI meant for transparency/reproducibility review. It **never calls any model APIs** and **only reads local artifacts** from `../results` (and optionally `../data`).

### Install (dashboard-only deps)

From the repo root:

```bash
pip install -r dashboard/requirements.txt
```

### Run

From the repo root:

```bash
streamlit run dashboard/app.py
```

## Figures (paper-ready export)

The **Figures** tab renders paper-ready matplotlib plots directly from:
- `results/scaling_results.csv` (or `*_clean.csv` / `*_raw.csv` if selected in the sidebar)
- `results/bootstrap_ci_*.json` (optional; for paired-diff CI)

It provides filters for `run_id`, `dataset`, and `method` (and optional `seed`) and can optionally overlay seeds.

### Export

Click **Export current figures** to write images to:
- `results/figures/`

Filenames include a timestamp and the selected filters, e.g.:
- `YYYYMMDD-HHMMSS_<dataset>_<method>_<run_id>_accuracy.png`
- `..._tokens.png`
- `..._caphit.png` (only if available)
- `..._diffci.png` (only if a bootstrap CI file + matching group exists)

Optionally enable **Also export SVG** in the Figures tab.

### Expected files (under `results/`)

- `results/scaling_results.csv` — written by `main.py` / `results_logger.py`
- `results/per_problem_results.jsonl` — written by `main.py` (used indirectly to produce bootstrap CIs)
- `results/bootstrap_ci_<run_id>_<metric>.json` — written by `scripts/bootstrap_ci.py`
- `results/extraction_audit_<timestamp>.csv` — written by `scripts/audit_extraction.py`

The Extraction Audit tab can export a **new** labeled CSV (`results/extraction_audit_labeled_*.csv`) and will never overwrite existing files.




