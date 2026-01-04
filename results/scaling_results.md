# Scaling Results

## Status

This document summarizes scaling experiment observations.
All tables are backed by committed snapshot files in `results/snapshots/`.

**Primary metric**: pass_at_k_accuracy  
**Compute unit**: decoded completion tokens

---

## Methodology

- **Model**: gpt-4o-mini
- **Temperature**: 0.7 for sampling
- **Budget**: `budget_tokens = multiplier × BASE_BUDGET_TOKENS` (BASE_BUDGET_TOKENS = 1004)
- **Seed**: Derived per-draw via sha256 for independence

---

## Results

*Note: Tables below require committed snapshots. If no snapshots exist, run experiments and create a snapshot:*

```bash
python main.py --dataset-path data/aime.jsonl --limit 50 --multipliers 1,10,100
python scripts/make_snapshot.py
```

### AIME Results

| Multiplier | pass@k (uniform) | pass@k (two-phase) |
|------------|------------------|-------------------|
| 1×         | ...              | ...               |
| 10×        | ...              | ...               |
| 100×       | ...              | ...               |

**Evidence**: See `results/snapshots/<run_id>_*` for committed CSV and manifest.

---

## Key observations

- Gains from 1×→10× are typically observable
- 10×→100× often shows diminishing returns
- Two-phase budgeted sampling does not consistently outperform uniform in tested runs

---

## Limitations

- **Extraction noise**: Answer parsing can introduce false negatives/positives
- **Small N**: Bootstrap CIs quantify uncertainty but don't eliminate dataset bias
- **Single model/prompt**: Results are specific to this configuration

---

## Non-claims

- No claim of general scaling laws
- No claim that two-phase sampling is superior to uniform
- Results are empirical observations for this harness/model/dataset combination

