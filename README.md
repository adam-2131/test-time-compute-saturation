# test-time-compute-saturation

Code and data for my Maturitätsarbeit (Kantonsschule Zürich Nord, 2026):
*Test-Time Compute Scaling through Parallel Sampling – A Controlled Empirical
Study of Saturation in Self-Consistency.*

The question: if you ask a language model the same math problem k times and
take the most frequent answer (majority voting / self-consistency), how much
does accuracy improve as k grows from 1 to 1000?

## Results

300 problems per run, same subset for every k.

| k | AIME, gpt-4o-mini | GSM8K, gpt-4o-mini | AIME, gpt-4.1-mini |
|---:|---:|---:|---:|
| 1 | 21.3 % | 91.3 % | 24.7 % |
| 10 | 26.3 % | 93.3 % | 31.0 % |
| 100 | 27.0 % | 93.7 % | 35.7 % |
| 1000 | 26.7 % | 93.7 % | 36.0 % |

Almost all of the gain happens between k = 1 and k = 10 (for gpt-4.1-mini up
to k = 100). After that, more samples cost a lot more and change nothing
measurable. With early stopping, a budget of k = 1000 used on average only
74 samples per problem on AIME and 18 on GSM8K.

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

An OpenAI API key is only needed for answers that are not in the cache yet:

```bash
export OPENAI_API_KEY="..."        # macOS/Linux
$env:OPENAI_API_KEY="..."          # Windows PowerShell
```

## Running the experiments

```bash
# AIME, gpt-4o-mini
python main.py --method fixed_k_majority \
    --dataset-path data/aime.jsonl --model gpt-4o-mini \
    --per-example-dir experiments

# GSM8K, gpt-4o-mini
python main.py --method fixed_k_majority \
    --dataset-path data/gsm8k.jsonl --model gpt-4o-mini \
    --per-example-dir experiments_gsm8k

# AIME, gpt-4.1-mini
python main.py --method fixed_k_majority \
    --dataset-path data/aime.jsonl --model gpt-4.1-mini \
    --per-example-dir experiments_aime_41mini
```

Each run writes one CSV per k value (`k_1.csv`, `k_10.csv`, ...) with the
columns

```
example_id,k_max,k_used,prediction,confidence,entropy,correct,runtime_seconds
```

Main options (defaults in brackets):

| Option | Default | Meaning |
|---|---|---|
| `--k-values` | 1,10,100,1000 | sample budgets to run |
| `--subset-size` | 300 | number of problems |
| `--subset-seed` | 123 | seed for choosing the subset |
| `--seed` | 0 | base seed, every API call gets its own seed derived from it |
| `--early-stop-confidence` | 0.95 | stop when the majority answer has this share of the votes |
| `--early-stop-stability` | 50 | stop when the majority answer hasn't changed for this many samples |
| `--early-stop-min-samples` | 10 | number of samples before either rule is checked |
| `--max-workers` | 32 | problems processed in parallel |

Temperature (0.7) and the token limit (2048) are fixed in `main.py`.

## Plots and statistics

```bash
python scripts/plot_frontier.py                 # plots/ for the AIME run
python scripts/plot_frontier.py --experiments-dir experiments_gsm8k --plots-dir plots_gsm8k
python scripts/plot_comparisons.py              # plots_combined/
python scripts/significance.py                  # McNemar test between neighbouring k
python scripts/explain_dip.py                   # problems that flip between k = 100 and 1000
```

## Tests

```bash
python -m pytest tests
```

The tests replace the API with a stub, so they run without a key.

## Files

```
main.py          runs the experiment
eval.py          sampling loop, majority vote, early stopping
utils.py         OpenAI API call with cache and retries
parse.py         extracts the final answer from a response
prompting.py     the prompt
data.py          loads a .jsonl problem set
scripts/         plots, statistics, building the datasets
tests/           pytest tests
data/            aime.jsonl (933 problems), gsm8k.jsonl (1319 problems)
experiments*/    results of the three runs
plots*/          figures
```

## Data

- AIME: [gneubig/aime-1983-2024](https://huggingface.co/datasets/gneubig/aime-1983-2024), rebuilt with `scripts/build_aime_jsonl.py`
- GSM8K: [gsm8k](https://huggingface.co/datasets/gsm8k), test split, rebuilt with `scripts/build_gsm8k_jsonl.py`

## Limitations

- Only two small models from the same provider.
- Answers are compared as text, so `025` and `25` (or `4.00` and `4`) count as
  different. This affects one AIME problem and ten GSM8K problems at every k,
  so it shifts the curves but doesn't change their shape.
- Answer extraction falls back to the last integer in the response if no
  final answer is marked, which can pick the wrong number.

## Note

Parts of the code were written with the help of an AI coding assistant. The
experiment design, the runs and the analysis are my own; see the "Tools Used"
section of the thesis.

## License

MIT
