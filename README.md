# Parallel-Sampling

Small research codebase for **test-time compute** experiments: allocate a fixed **decoded-token budget** per problem, run repeated sampling, and measure **pass@k / accuracy vs compute**.

This repo also includes a tiny demo of **Pydantic-validated structured sampling** (`idea.py`), but the main focus is the math scaling harness (`main.py`).

## What it does

For each problem:

- Run a **greedy baseline** (temperature 0) to estimate typical completion tokens.
- Pick a compute multiplier \(m \in \{1,10,100,1000\}\) and define a per-problem budget:
  - \( \text{budget_tokens} = m \times \text{baseline_avg_completion_tokens} \)
- Keep sampling solutions (temperature > 0) until the budget would be exceeded.
- Mark the problem as **pass** if any sampled solution matches the gold answer after extraction.

Results are appended to `results/scaling_results.csv` (ignored by git by default). Curated/sanity-check artifacts live in `results/`.

## Quickstart

### 1) Install

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

### 2) Set `OPENAI_API_KEY`

- **macOS/Linux**:

```bash
export OPENAI_API_KEY="..."
```

- **Windows PowerShell**:

```powershell
$env:OPENAI_API_KEY="..."
```

### 3) Run the scaling experiment

```bash
python main.py
```

By default this runs on `data/frontier_math.jsonl`, which is currently a **toy placeholder dataset** (2 arithmetic problems) matching the “sanity check” writeup in `results/scaling_results.md`.

## Report & Figures (no new model calls)

This repo includes a deterministic post-hoc bundler that turns existing result artifacts into a submission-ready folder (CSV snapshot, JSON summary, publication-quality figures, and an auto-generated executive summary):

```bash
python scripts/make_report_bundle.py
```

Outputs:
- `results/report_bundle/`
  - `fig1_accuracy_vs_compute.png`
  - `fig2_tokens_vs_compute.png`
  - `fig3_cap_hit_rate_vs_compute.png`
  - `fig4_paired_diff_ci_100x_minus_10x.png`
  - `executive_summary.md`, `summary.json`, `manifest.json`

### Run all post-hoc steps

If you want a single command that (1) cleans/rebuilds CSVs, (2) runs bootstrap CIs if possible, and (3) generates the report bundle:

```bash
# macOS/Linux (or Git-Bash)
bash scripts/run_all_posthoc.sh

# Windows PowerShell
powershell -ExecutionPolicy Bypass -File scripts/run_all_posthoc.ps1
```

## Paper (Maturaarbeit-ready PDF source)

The formal report lives in `paper/` (LaTeX). It embeds figures directly from `results/report_bundle/`.

Build steps (no model calls):

```bash
python scripts/make_report_bundle.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Output: `paper/build/main.pdf`

## Data format

`data/frontier_math.jsonl` is JSONL with one object per line:

```json
{"id":"0","problem":"...","answer":"..."}
```

If you swap in a real dataset, keep the same keys (`id`, `problem`, `answer`).

## Outputs

- **Caching**: model responses are cached in `.cache/` (ignored by git).
- **Results log**: `results/scaling_results.csv` (auto-created, append-only).
- **Curated artifacts**: `results/scaling_results.md`, `results/scaling_results_raw.csv`, `results/scaling_results_clean.csv`.
  - To clean legacy/raw logs:

```bash
python scripts/clean_results.py
```

### Results & Audit Dashboard (read-only)

To inspect runs and audit artifacts locally (no API calls; reads only from `./results`):

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Expected files under `results/`:

- `scaling_results.csv` — written by `main.py` / `results_logger.py`
- `per_problem_results.jsonl` — written by `main.py`
- `bootstrap_ci_<run_id>_<metric>.json` — written by `scripts/bootstrap_ci.py`
- `extraction_audit_<timestamp>.csv` — written by `scripts/audit_extraction.py`

- **Bootstrap CIs (no new model calls)**: after running `main.py` at least once, compute bootstrap 95% CIs over problems from `results/per_problem_results.jsonl`:

```bash
python scripts/bootstrap_ci.py
```

- **Answer extraction audit (no new model calls)**: sample cached model outputs and export a CSV for manual annotation:

```bash
python scripts/audit_extraction.py --n 50
```

## Reproducibility

### Run the main experiments

Uniform sampling (budgeted best-of-N):

```bash
python main.py --dataset-path data/aime.jsonl --dataset-name aime --limit 50 --multipliers 1,10,100 --method uniform_best_of_n_budgeted --seed 0
```

Two-stage self-consistency (alias: `successive_halving_budgeted`):

```bash
python main.py --dataset-path data/aime.jsonl --dataset-name aime --limit 50 --multipliers 1,10,100 --method two_stage_self_consistency --seed 0
```

To verify the exact per-problem budgets used:

```bash
python main.py --debug-budget --dataset-path data/aime.jsonl --dataset-name aime --limit 50 --multipliers 1,10,100 --method uniform_best_of_n_budgeted --seed 0
```

### How compute budgets are defined

- **Budget unit**: decoded **completion tokens**.
- **Per-problem budget**: `budget_tokens = compute_multiplier * BASE_BUDGET_TOKENS` (see `main.py`; current default `BASE_BUDGET_TOKENS = 1004`).
- **Budget enforcement**: evaluation loops stop sampling when adding the next completion would exceed `budget_tokens` (logging also tracks total tokens separately).

### Regenerate tables / artifacts (no new model calls)

- **Clean / rebuild CSVs** (for plotting/reporting):

```bash
python scripts/clean_results.py
python scripts/rebuild_scaling_results.py
```

- **Bootstrap 95% CIs over problems** (writes `results/bootstrap_ci_<run_id>_<metric>.json`):

```bash
python scripts/bootstrap_ci.py --metric accuracy --diff 10,100 --n-bootstrap 10000 --seed 0
```

## Repo layout

- `main.py`: runs the budgeted best-of-n experiment and prints a small table.
- `eval.py`: implements the per-problem sampling loop under a decoded-token budget.
- `parse.py`: extracts “Final Answer: ...” / boxed answers from model text.
- `prompting.py`: prompt formatting for math problems.
- `results_logger.py`: appends rows and enforces a stable CSV schema.
- `utils.py`: OpenAI call helper (`call_model`) + minimal Pydantic JSON sampling (`sample`).
- `idea.py`: structured sampling demo (DALY idea generation).

## Notes / gotchas

- This repo uses `openai`’s `chat.completions` API.
- `utils.call_model()` caches by `(model, temperature, max_tokens, prompt, seed)` (and optional `cache_salt`). If you change prompts or params, you’ll get new cache entries.
- Answer extraction is intentionally minimal (`parse.py`); for harder datasets you may want to tighten normalization.
## Project scope & limitations (explicit)

- **No claim of general scaling laws.** Any observed curves are empirical for this harness, dataset(s), prompt, and model configuration.
- **Single model family and limited datasets/subsets.** Results may not transfer to other models or domains.
- **Extraction is a known failure mode.** The repo includes an extraction audit tool, but extraction noise can still affect results.
- **Token budgets are an approximation of compute.** This project focuses on decoded completion tokens as the budget unit.

