# AutoGluon Report — titanic

**Date:** 2026-06-02 19:58
**Preset:** `medium_quality` &nbsp;|&nbsp; **Time limit:** 120 s

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
| Approx. execution time | ~22.2 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **0.7821** |
| F1 (weighted) | **0.7821** |

---

## Leaderboard (Top 5)

| # | Model | Score (F1-weighted) | Fit time (s) |
|---|-------|--------------|-------------|
| 1 | `NeuralNetTorch` | 0.8142 | 4.6 |
| 2 | `CatBoost` | 0.7898 | 0.8 |
| 3 | `LightGBMLarge` | 0.7871 | 1.5 |
| 4 | `NeuralNetFastAI` | 0.7847 | 1.7 |
| 5 | `LightGBMXT` | 0.7821 | 3.5 |

> Scores computed on the held-out test split using **f1_weighted** as the ranking metric.
