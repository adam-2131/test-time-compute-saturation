# Parallel-Sampling

Research codebase for **test-time compute scaling** experiments: measure how accuracy scales with parallel sampling on math problems.

**Core claim under investigation:** Test-time compute scaling via parallel sampling saturates after ~10× and yields diminishing accuracy gains.

## Quickstart

### 1) Install

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

### 2) Set `OPENAI_API_KEY`

```bash
# macOS/Linux
export OPENAI_API_KEY="..."

# Windows PowerShell
$env:OPENAI_API_KEY="..."
```

### 3) Run the fixed-k experiment

```bash
python main.py --method fixed_k_majority
```

This runs the primary experiment:
- Samples k ∈ {1, 10, 100, 1000} completions per problem
- Uses majority voting for aggregation
- Applies early stopping (confidence ≥ 0.95 or stable majority for 50 samples)
- Outputs per-example CSVs to `experiments/k_*.csv`

### 4) Generate plots

```bash
python scripts/plot_frontier.py
```

Produces:
- `plots/accuracy_vs_k.png` — Accuracy vs compute (log scale)
- `plots/marginal_gain.png` — Marginal gains per 10× compute
- `plots/effective_compute.png` — Early stopping savings

## Experiment Details

### Fixed-k Majority Voting (Primary)

```bash
python main.py --method fixed_k_majority \
    --k-values 1,10,100,1000 \
    --subset-size 300 \
    --subset-seed 123 \
    --early-stop-stability 50 \
    --early-stop-confidence 0.95
```

**Output format** (`experiments/k_*.csv`):
```
example_id,k_max,k_used,prediction,confidence,entropy,correct,runtime_seconds
```

### Key parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--k-values` | 1,10,100,1000 | Sample counts to evaluate |
| `--subset-size` | 300 | Fixed evaluation subset size |
| `--subset-seed` | 123 | Seed for deterministic subset selection |
| `--early-stop-stability` | 50 | Stop if majority unchanged for N samples |
| `--early-stop-confidence` | 0.95 | Stop if confidence ≥ threshold |

## Repository Structure

```
├── main.py              # Experiment runner
├── eval.py              # Sampling and aggregation logic
├── data.py              # Dataset loading
├── parse.py             # Answer extraction
├── prompting.py         # Prompt formatting
├── utils.py             # OpenAI API wrapper with caching
│
├── scripts/
│   ├── plot_frontier.py     # Generate accuracy/compute plots
│   └── build_aime_jsonl.py  # Build dataset from HuggingFace
│
├── data/
│   └── aime.jsonl       # AIME math problems (933 problems)
│
├── experiments/         # Per-example CSVs (generated)
│   └── k_*.csv
│
└── plots/               # Figures (generated)
    ├── accuracy_vs_k.png
    ├── marginal_gain.png
    └── effective_compute.png
```

## Data Format

`data/aime.jsonl` contains one JSON object per line:

```json
{"id": "1983-1", "problem": "Let x, y and z...", "answer": "60"}
```

To rebuild from HuggingFace:
```bash
pip install datasets
python scripts/build_aime_jsonl.py
```

## Caching

Model responses are cached in `.cache/` (gitignored). Cache keys include `(model, temperature, max_tokens, prompt, seed)`. Delete `.cache/` to force fresh API calls.

## Legacy Methods

The repo also contains legacy token-budgeted experiments (not the primary focus):

```bash
# Token-budgeted uniform sampling
python main.py --method uniform_best_of_n_budgeted --multipliers 1,10,100

# Token-budgeted two-phase sampling  
python main.py --method two_phase_budgeted_sampling --multipliers 1,10,100
```

These write to `results/scaling_results.csv` and `results/per_problem_results.jsonl`.

## Limitations

- **Single model family**: Results are specific to gpt-4o-mini
- **Extraction noise**: Answer parsing can introduce errors
- **No claim of general scaling laws**: Empirical observations only

## License

MIT
