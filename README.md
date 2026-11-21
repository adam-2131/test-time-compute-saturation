# test-time-compute-saturation

Code and data for my Maturitätsarbeit (Kantonsschule Zürich Nord, 2026):
*Test-Time Compute Scaling through Parallel Sampling – A Controlled Empirical
Study of Saturation in Self-Consistency.*

The question: if you ask a language model the same math problem k times and
take the most frequent answer (majority voting / self-consistency), how much
does accuracy improve as k grows from 1 to 1000?

## Results

Final runs: 300 problems per run, same subset for every k, token limit 16384.

| k | AIME, gpt-4o-mini | GSM8K, gpt-4o-mini | AIME, gpt-4.1-mini |
|---:|---:|---:|---:|
| 1 | 20.00 % | 94.00 % | 64.67 % |
| 10 | 25.00 % | 97.00 % | 76.33 % |
| 100 | 27.67 % | 97.00 % | 78.00 % |
| 1000 | 27.67 % | 97.00 % | 78.00 % |

In every run the only statistically significant gain is from k = 1 to k = 10
(exact McNemar test, Holm-corrected). With early stopping, a budget of
k = 1000 used on average 72.7 (gpt-4o-mini) and 44.2 (gpt-4.1-mini) samples
per problem on AIME and 15.9 on GSM8K.

The full statistical analysis (validation, significance tests, token usage,
figures) is in
[parallel-sampling-analysis](https://github.com/adam-2131/parallel-sampling-analysis).
The `experiments*/` folders in this repository contain the earlier runs with a
2048-token limit and the original parser; they are kept for reference and are
not comparable with the final runs.

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

The final runs used these commands:

```bash
# AIME, gpt-4o-mini
python main.py --method fixed_k_majority --max-tokens 16384 \
    --dataset-path data/aime.jsonl --model gpt-4o-mini \
    --per-example-dir experiments_aime_4o_16k --max-workers 48

# GSM8K, gpt-4o-mini
python main.py --method fixed_k_majority --max-tokens 16384 \
    --dataset-path data/gsm8k.jsonl --model gpt-4o-mini \
    --per-example-dir experiments_gsm8k_16k --max-workers 32

# AIME, gpt-4.1-mini
python main.py --method fixed_k_majority --max-tokens 16384 \
    --dataset-path data/aime.jsonl --model gpt-4.1-mini \
    --per-example-dir experiments_aime_41mini_16k --max-workers 64
```

Long responses need a longer request timeout, e.g.
`$env:OPENAI_REQUEST_TIMEOUT_S="600"`.

Each run writes one CSV per k value (`k_1.csv`, `k_10.csv`, ...) with the
columns

```
example_id,k_max,k_used,valid_votes,invalid_votes,token_limit_votes,prediction,confidence,entropy,correct,completion_tokens,runtime_seconds
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
| `--early-stop-min-valid` | 10 | number of valid answers before either rule can stop sampling |
| `--max-tokens` | 2048 | token limit per response (the final runs used 16384) |
| `--limit` | – | only run the first N problems of the subset (pilot runs) |
| `--max-workers` | 32 | problems processed in parallel |

Temperature (0.7) is fixed in `main.py`. Responses without a valid final
answer count as samples but never move the vote towards stopping. The samples
of one problem are drawn one after another so that early stopping can be
checked after each response; `--max-workers` parallelises different problems.

`scripts/replay_corrected.py` re-scores the earlier 2048-token runs offline
from the response cache with the current parser (no API calls).

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
experiments*/    earlier 2048-token runs (reference only)
plots*/          figures of the earlier runs
```

## Data

- AIME: [gneubig/aime-1983-2024](https://huggingface.co/datasets/gneubig/aime-1983-2024), rebuilt with `scripts/build_aime_jsonl.py`
- GSM8K: [gsm8k](https://huggingface.co/datasets/gsm8k), test split, rebuilt with `scripts/build_gsm8k_jsonl.py`

The files in `data/` are the exact versions the experiments used. The current
Hugging Face copy of AIME differs from it by one double space (problem 1984-1).

## Limitations

- Only two small models from the same provider.
- Answer extraction is conservative: only an explicit "Final Answer:" label
  or a `\boxed{}` counts, so a correct answer without either is scored as
  invalid. This affected under 1 % of the final responses.
- The AIME problems (1983–2024) were public before both models were
  released, so training-data contamination cannot be ruled out.
- No run without early stopping exists, so accuracy at a full k = 1000
  budget is not measured.

## Note

Parts of the code were written with the help of an AI coding assistant. The
experiment design, the runs and the analysis are my own; see the "Tools Used"
section of the thesis.

## License

MIT
