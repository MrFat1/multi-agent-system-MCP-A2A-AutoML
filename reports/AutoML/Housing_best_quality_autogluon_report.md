# AutoGluon Report — Housing

**Date:** 2026-06-02 20:23
**Preset:** `best_quality` &nbsp;|&nbsp; **Time limit:** 120 s

---

## Dataset

| Field | Value |
|-------|-------|
| Name | `Housing` |
| Task | Regression |
| Target column | `price` |
| Train samples | 436 |
| Test samples | 109 |

---

## Best Model

| Field | Value |
|-------|-------|
| Model | `WeightedEnsemble_L3` |
| Approx. execution time | ~135.0 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| R² | **0.6290** |
| RMSE | **1369331.5804** |

---

## Leaderboard (Top 5)

| # | Model | Score (R²) | Fit time (s) |
|---|-------|--------------|-------------|
| 1 | `WeightedEnsemble_L2` | 0.6297 | 10.8 |
| 2 | `WeightedEnsemble_L3` | 0.6290 | 26.4 |
| 3 | `RandomForestMSE_BAG_L1` | 0.6284 | 1.5 |
| 4 | `CatBoost_BAG_L1` | 0.6217 | 8.2 |
| 5 | `LightGBMXT_BAG_L2` | 0.6199 | 26.4 |

> Scores computed on the held-out test split using **r2** as the ranking metric.

(_ray_fit pid=7592) [1000]      valid_set's l2: 7.74761e+11     valid_set's r2: 0.747015
(_ray_fit pid=7592)     Ran out of time, early stopping on iteration 1013. Best iteration is:
(_ray_fit pid=7592)     [763]   valid_set's l2: 7.6213e+11      valid_set's r2: 0.75114
