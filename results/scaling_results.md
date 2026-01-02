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


