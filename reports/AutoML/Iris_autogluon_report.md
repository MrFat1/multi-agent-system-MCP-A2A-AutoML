# AutoGluon Report — Iris

**Date:** 2026-06-02 20:09
**Preset:** `best_quality` &nbsp;|&nbsp; **Time limit:** 120 s

---

## Dataset

| Field | Value |
|-------|-------|
| Name | `Iris` |
| Task | Multiclass Classification |
| Target column | `Species` |
| Train samples | 117 |
| Test samples | 30 |

---

## Best Model

| Field | Value |
|-------|-------|
| Model | `WeightedEnsemble_L3` |
| Approx. execution time | ~138.7 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **0.9333** |
| F1 (weighted) | **0.9333** |

---

## Leaderboard (Top 5)

| # | Model | Score (F1-weighted) | Fit time (s) |
|---|-------|--------------|-------------|
| 1 | `CatBoost_BAG_L1` | 0.9333 | 4.0 |
| 2 | `LightGBMXT_BAG_L1` | 0.9333 | 4.8 |
| 3 | `RandomForestEntr_BAG_L1` | 0.9333 | 0.6 |
| 4 | `RandomForestGini_BAG_L1` | 0.9333 | 1.0 |
| 5 | `NeuralNetFastAI_BAG_L2` | 0.9333 | 48.2 |

> Scores computed on the held-out test split using **f1_weighted** as the ranking metric.
