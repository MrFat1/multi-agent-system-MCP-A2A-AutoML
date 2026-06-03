# AutoGluon Report — Housing

**Date:** 2026-06-02 20:00
**Preset:** `medium_quality` &nbsp;|&nbsp; **Time limit:** 120 s

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
| Model | `WeightedEnsemble_L2` |
| Approx. execution time | ~19.1 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| R² | **0.6260** |
| RMSE | **1374993.5162** |

---

## Leaderboard (Top 5)

| # | Model | Score (R²) | Fit time (s) |
|---|-------|--------------|-------------|
| 1 | `WeightedEnsemble_L2` | 0.6260 | 6.2 |
| 2 | `NeuralNetTorch` | 0.6082 | 4.8 |
| 3 | `RandomForestMSE` | 0.6069 | 0.7 |
| 4 | `CatBoost` | 0.6004 | 1.8 |
| 5 | `LightGBM` | 0.5925 | 0.6 |

> Scores computed on the held-out test split using **r2** as the ranking metric.
