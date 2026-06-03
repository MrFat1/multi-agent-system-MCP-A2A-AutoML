# AutoGluon Report — titanic

**Date:** 2026-06-02 20:17
**Preset:** `best_quality` &nbsp;|&nbsp; **Time limit:** 120 s

---

## Dataset

| Field | Value |
|-------|-------|
| Name | `titanic` |
| Task | Binary Classification |
| Target column | `Survived` |
| Train samples | 624 |
| Test samples | 156 |

---

## Best Model

| Field | Value |
|-------|-------|
| Model | `WeightedEnsemble_L2` |
| Approx. execution time | ~138.8 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **0.7885** |
| F1 (weighted) | **0.7838** |

---

## Leaderboard (Top 5)

| # | Model | Score (F1-weighted) | Fit time (s) |
|---|-------|--------------|-------------|
| 1 | `NeuralNetFastAI_BAG_L1` | 0.7959 | 20.1 |
| 2 | `CatBoost_BAG_L1` | 0.7898 | 7.4 |
| 3 | `LightGBMXT_BAG_L2` | 0.7888 | 41.6 |
| 4 | `LightGBM_BAG_L1` | 0.7838 | 5.8 |
| 5 | `WeightedEnsemble_L2` | 0.7838 | 6.7 |

> Scores computed on the held-out test split using **f1_weighted** as the ranking metric.
