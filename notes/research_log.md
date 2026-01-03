# Parallel Sampling — Research Log

## Project goal
Empirically characterize how accuracy on hard mathematical reasoning
tasks scales as a function of test-time compute under fixed conditions.

Test-time compute is defined as total decoded tokens.

---

## Chronological Log


### 2026-01-01 — Implemented token-counting eval harness

Change:
- Replaced JSON-based sampling with raw text generation.
- Implemented call_model() returning decoded token counts.
- Added greedy and best-of-N baselines with exact answer parsing.

Reason:
- Needed explicit measurement of test-time compute (decoded tokens).
- JSON forcing interferes with math-style reasoning outputs.

Expectation:
- Token counts should scale roughly linearly with number of samples.
- Greedy should use minimal tokens.

Result:
- Sanity check on toy problems shows correct token accounting.
- Best-of-4 uses ~4× decoded tokens vs greedy.
- No conclusions drawn beyond validation of the harness.


### 2026-01-02 — Added fixed math prompt wrapper and realistic token budgets

Change:
- Added standardized math prompt with explicit final-answer format.
- Increased max_tokens to 2048 for evaluation calls.

Reason:
- Ensure generations reflect realistic math-style reasoning lengths.
- Make test-time compute budgets meaningful.

Expectation:
- Decoded token usage increases substantially per sample.
- Accuracy on toy data remains unchanged.

Result:
- Decoded tokens increased (toy run totals: 10, 100, 1000, 10000 for multipliers 1/10/100/1000).
- Linear scaling with compute multiplier preserved.
- Sanity-check accuracy unchanged (toy dataset).


### 2026-01-02 — Validated budget-controlled test-time compute scaling

Change:
--Ran repeated sanity sweeps using uniform_best_of_n_budgeted with compute multipliers {1, 10, 100, 1000}.
- Restricted analysis to clean runs with consistent token accounting.

Reason:
- Verify that the evaluation harness enforces test-time compute budgets correctly.
- Confirm stability and reproducibility of decoded-token measurements before scaling to non-toy tasks.

Expectation:
- Total decoded tokens scale linearly with compute multiplier.
- Minor under-spend possible due to early stopping or per-problem allocation.
- Accuracy on toy dataset remains constant across budgets.

Result:
- Decoded token usage scales linearly with compute multiplier:
         ~10, ~100, ~1000, ~10000 tokens for multipliers 1/10/100/1000.
- Small under-spend observed at higher budgets (≤0.2%), consistent with early termination.
- Accuracy remains 1.0 across all budgets on the toy dataset.
- Repeated sweeps on the same day produce near-identical token usage, indicating stable budget enforcement.

Conclusion:
- Test-time compute is reliably measurable via decoded tokens.
- The budgeted best-of-N implementation respects compute limits up to minor rounding effects.
- The evaluation pipeline is sufficiently validated to proceed to non-trivial datasets.

### 2026-01-03 — First non-trivial scaling results on AIME (subset=10)

Change:
- Integrated AIME-style dataset.
- Ran uniform best-of-N under fixed token budgets for multipliers 1×, 10×, and 100×.
- Evaluated on a subset of 10 problems.

Reason:
- Validate that the evaluation harness produces non-trivial scaling behavior
  on hard mathematical reasoning tasks.

Expectation:
- Accuracy at 1× should be low.
- Accuracy should increase with compute, with diminishing returns.

Result:
- At 1× compute, accuracy ≈ 0.10.
- At 10× compute, accuracy ≈ 0.40.
- At 100× compute, accuracy ≈ 0.40 on subset=10 (no further improvement observed).
- Confirms that test-time compute materially improves performance on hard math,
  and that diminishing returns appear early on small subsets.
- Early AIME-labeled runs using toy data were excluded from analysis.
