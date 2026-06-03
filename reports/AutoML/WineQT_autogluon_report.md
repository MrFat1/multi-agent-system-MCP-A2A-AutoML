# AutoGluon Report — WineQT

**Date:** 2026-06-02 20:06
**Preset:** `medium_quality` &nbsp;|&nbsp; **Time limit:** 120 s

---

## Dataset

| Field | Value |
|-------|-------|
| Name | `WineQT` |
| Task | Multiclass Classification |
| Target column | `quality` |
| Train samples | 814 |
| Test samples | 204 |

---

## Best Model

| Field | Value |
|-------|-------|
| Model | `WeightedEnsemble_L2` |
| Approx. execution time | ~27.1 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **0.5245** |
| F1 (weighted) | **0.4879** |

---

## Leaderboard (Top 5)

| # | Model | Score (F1-weighted) | Fit time (s) |
|---|-------|--------------|-------------|
| 1 | `CatBoost` | 0.5535 | 2.1 |
| 2 | `NeuralNetTorch` | 0.5446 | 4.6 |
| 3 | `RandomForestGini` | 0.5432 | 0.7 |
| 4 | `LightGBMXT` | 0.5375 | 1.0 |
| 5 | `RandomForestEntr` | 0.5301 | 0.6 |

> Scores computed on the held-out test split using **f1_weighted** as the ranking metric.
