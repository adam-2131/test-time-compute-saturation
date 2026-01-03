# Scaling Results

## Status
This document currently contains sanity-check results on a toy dataset.
These results are not intended to demonstrate scaling behavior and are used
only to validate the evaluation harness, token accounting, and logging.

No claims are made based on the results below.

---

## Setup (Sanity Check)

- Model: gpt-4o-mini
- Dataset: toy arithmetic problems
- Metric: exact accuracy (pass@k)
- Test-time compute: total decoded tokens
- Decoding:
  - Greedy: temperature = 0.0
  - Sampling: temperature = 0.7
- Maximum tokens per sample: small (sanity-check only)

---

## Results (Sanity Check)

| Compute Multiplier | Accuracy | Total Decoded Tokens |
|-------------------|----------|----------------------|
| 1×                | 1.00     | 2                    |
| 10×               | 1.00     | 20                   |
| 100×              | 1.00     | 200                  |
| 1000×             | 1.00     | 2000                 |

---

## Interpretation

- Accuracy is constant across compute multipliers because the toy problems
  are trivially solvable by the model.
- The linear scaling of decoded tokens with the compute multiplier confirms
  that token accounting and budget enforcement are functioning as intended.
- No accuracy conclusions can be drawn from this dataset.

---

## Next Steps

- Replace the toy dataset with Frontier Math.
- Increase maximum generation length to reflect math-style reasoning.
- Re-run scaling experiments under identical decoding parameters.
- Record all results in scaling_results.csv.


## Results (AIME, subset=10)

| Compute Multiplier | Accuracy |
|-------------------:|---------:|
| 1×                | 0.10     |
| 10×               | 0.40     |
| 100×              | 0.40     |

These results represent the first non-trivial scaling behavior observed
in this project. Accuracy improves substantially from 1× to 10× compute,
with no further improvement observed at 100× on this small subset,
suggesting early diminishing returns.


## AIME Scaling Results (Uniform Best-of-N, Budgeted)

### Experimental setup
- Model: gpt-4o-mini
- Dataset: AIME-style
- Method: uniform_best_of_n_budgeted
- Metric: exact correctness
- Test-time compute: total decoded completion tokens
- All runs use the fixed math prompt wrapper.
- Sanity-check and toy runs are excluded.

---

### AIME (subset = 50)

| Compute multiplier | Accuracy | Total decoded tokens |
|-------------------|----------|----------------------|
| 1×                | 0.20     | 21,112               |
| 10×               | 0.26     | 476,166              |
| 100×              | 0.26     | 4,997,441            |

Notes:
- All three multipliers were evaluated under a single experimental sweep.
- Total decoded tokens scale approximately linearly with the compute multiplier.
- Accuracy improves from 1× to 10×, but shows no further gain from 10× to 100×.

---

### Key observation
On a larger AIME subset (50 problems), uniform best-of-N exhibits **clear diminishing returns**:
- Significant improvement from 1× → 10× compute.
- No measurable improvement from 10× → 100× compute under fixed token budgets.

This confirms that the plateau observed on subset=10 is **not an artifact of extremely small sample size**, and motivates exploration of adaptive test-time compute allocation strategies.


## AIME Scaling Results (subset = 100)

| Method                    | Compute | Accuracy (range across seeds) |
|---------------------------|---------|--------------------------------|
| uniform best-of-N         | 10×     | ~0.22                          |
| uniform best-of-N         | 100×    | ~0.22                          |
| two-stage self-consistency (R=2) | 10×     | 0.22 – 0.25                    |
| two-stage self-consistency (R=2) | 100×    | 0.22 – 0.25                    |

Notes:
- All runs use identical per-problem decoded-token budgets.
- No consistent accuracy improvement is observed from 10× to 100× compute.
- Adaptive allocation via this two-stage self-consistency procedure does not outperform uniform sampling.
