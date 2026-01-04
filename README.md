# Parallel-Sampling

Small research codebase for **test-time compute** experiments: allocate a fixed **decoded-token budget** per problem, run repeated sampling, and measure **pass@k accuracy vs compute**.

## What it does

For each problem:

- Define a per-problem budget in **decoded completion tokens**:
  - `budget_tokens = multiplier × BASE_BUDGET_TOKENS` (fixed, no API jitter)
  - Current default: `BASE_BUDGET_TOKENS = 1004`
- Keep sampling solutions (temperature > 0) until the budget would be exceeded.
- **Primary metric (pass@k)**: Did any sampled solution match the gold answer?
- **Secondary metric (majority vote)**: Did the majority answer match the gold answer?

## Results artifacts

- **Local (gitignored)**: `results/scaling_results.csv` — generated at runtime
- **Committed**: `results/snapshots/` — curated snapshots with manifest

## Quickstart

### 1) Install

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

### 2) Set `OPENAI_API_KEY`

```bash
export OPENAI_API_KEY="..."   # macOS/Linux
$env:OPENAI_API_KEY="..."     # Windows PowerShell
```

### 3) Run an experiment

```bash
python main.py --dataset-path data/aime.jsonl --limit 50 --multipliers 1,10,100
```

### 4) Create a snapshot (for committing)

```bash
python scripts/make_snapshot.py
```

## Methodology

### Compute definition

- **Compute unit**: decoded **completion tokens** (not prompt tokens, not wall-clock time)
- **Per-problem budget**: `budget_tokens = compute_multiplier × BASE_BUDGET_TOKENS`
- **Budget enforcement**: sampling stops when adding next completion would exceed budget

### Primary metric: pass_at_k_accuracy

- **pass@k**: Did any of the k samples produce the correct answer?
- This metric is **consistent across all methods** (uniform and two-phase)
- The `pass_at_k_accuracy` column in results CSVs is the primary metric

### Sampling independence

Each model call uses a **derived seed** computed as:
```python
seed = int(sha256(f"{base_seed}|{problem_id}|{draw_idx}")[:8], 16)
```
This ensures each draw is independent while remaining deterministic.

### Methods

| Method | Description |
|--------|-------------|
| `uniform_best_of_n_budgeted` | Uniform sampling until budget exhausted |
| `two_phase_budgeted_sampling` | Two-phase budget schedule (see below) |

### Two-phase budgeted sampling (NOT successive halving)

This is a **budget schedule**, not adaptive allocation:

1. **Round 1**: Generate N=16 short samples, score by answer frequency
2. **Round 2**: Generate fresh samples until budget exhausted, majority vote

**Important**: Round 2 does NOT continue from round 1 traces. There is no actual "halving" or trace continuation. This is simply a budget allocation schedule.

Legacy aliases (emit warnings):
- `successive_halving_budgeted` → `two_phase_budgeted_sampling`
- `two_stage_self_consistency` → `two_phase_budgeted_sampling`

### Limitations

- **Extraction noise**: Answer parsing (`parse.py`) can introduce false negatives/positives
- **Token cap hit rate**: When completions hit `max_tokens`, effective compute may differ from nominal budget
- **Single model/prompt**: Results are specific to gpt-4o-mini and the prompt template
- **Small subsets**: For small N, variance is high; bootstrap CIs help quantify uncertainty

## Commands

### Run experiments

```bash
# Uniform sampling
python main.py --method uniform_best_of_n_budgeted --limit 50 --multipliers 1,10,100

# Two-phase sampling
python main.py --method two_phase_budgeted_sampling --limit 50 --multipliers 1,10,100
```

### Post-hoc analysis (no new model calls)

```bash
# Create snapshot for committing
python scripts/make_snapshot.py

# Rebuild curated CSV from raw
python scripts/rebuild_scaling_results.py

# Bootstrap 95% CIs
python scripts/bootstrap_ci.py --metric pass_at_k

# Audit answer extraction
python scripts/audit_extraction.py --n 50
```

## Repo layout

```
├── main.py              # Experiment runner
├── eval.py              # Per-problem sampling loops
├── parse.py             # Answer extraction
├── prompting.py         # Prompt formatting
├── data.py              # JSONL loading
├── results_logger.py    # CSV schema + logging
├── utils.py             # OpenAI helper + caching
├── data/                # JSONL datasets
├── results/
│   ├── snapshots/       # Committed artifacts (tracked)
│   └── scaling_results.csv  # Local log (gitignored)
├── scripts/
│   ├── make_snapshot.py     # Create committed snapshot
│   ├── rebuild_scaling_results.py
│   ├── bootstrap_ci.py
│   └── audit_extraction.py
```

## CSV schema

```
run_id,date,model,dataset,method,compute_multiplier,
pass_at_k_accuracy,majority_vote_accuracy,
total_decoded_tokens,total_completion_tokens_used,total_tokens_used,
num_problems,notes
```

Primary metric is `pass_at_k_accuracy`. No separate "accuracy" column.

## Non-claims

- No claim of general scaling laws
- No claim that two-phase sampling is superior to uniform
- Results are empirical observations for this specific setup
