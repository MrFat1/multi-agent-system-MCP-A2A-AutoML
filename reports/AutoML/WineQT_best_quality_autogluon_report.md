# AutoGluon Report — WineQT

**Date:** 2026-06-02 20:28
**Preset:** `best_quality` &nbsp;|&nbsp; **Time limit:** 120 s

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
| Approx. execution time | ~143.1 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **0.5735** |
| F1 (weighted) | **0.5428** |

---

## Leaderboard (Top 5)

| # | Model | Score (F1-weighted) | Fit time (s) |
|---|-------|--------------|-------------|
| 1 | `NeuralNetFastAI_BAG_L1` | 0.5428 | 17.5 |
| 2 | `WeightedEnsemble_L2` | 0.5428 | 17.6 |
| 3 | `WeightedEnsemble_L3` | 0.5428 | 17.8 |
| 4 | `LightGBMXT_BAG_L1` | 0.5407 | 9.0 |
| 5 | `NeuralNetFastAI_BAG_L2` | 0.5222 | 50.0 |

> Scores computed on the held-out test split using **f1_weighted** as the ranking metric.
