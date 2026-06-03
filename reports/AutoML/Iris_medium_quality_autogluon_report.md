# AutoGluon Report — Iris

**Date:** 2026-06-02 20:14
**Preset:** `medium_quality` &nbsp;|&nbsp; **Time limit:** 120 s

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
| Model | `WeightedEnsemble_L2` |
| Approx. execution time | ~19.0 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **0.9000** |
| F1 (weighted) | **0.8983** |

---

## Leaderboard (Top 5)

| # | Model | Score (F1-weighted) | Fit time (s) |
|---|-------|--------------|-------------|
| 1 | `NeuralNetTorch` | 0.9667 | 2.8 |
| 2 | `LightGBMXT` | 0.9346 | 0.8 |
| 3 | `ExtraTreesGini` | 0.8994 | 0.5 |
| 4 | `ExtraTreesEntr` | 0.8994 | 0.5 |
| 5 | `NeuralNetFastAI` | 0.8983 | 4.2 |

> Scores computed on the held-out test split using **f1_weighted** as the ranking metric.
