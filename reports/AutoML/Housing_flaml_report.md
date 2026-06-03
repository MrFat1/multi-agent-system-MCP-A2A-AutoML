# FLAML Report — Housing

**Date:** 2026-06-02 21:05
**Time limit:** 300 s

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
| Model | `xgboost` |
| Best validation loss | 0.3442 |
| Best iteration | 354 |
| Approx. execution time | ~299.8 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| R² | **0.6262** |
| RMSE | **1374541.2174** |

---

## Per-Estimator Leaderboard (Top 5)

| # | Model | Score (R²) | Val. loss |
|---|-------|--------------|-----------|
| 1 | `xgboost` | 0.6558 | 0.3442 |
| 2 | `catboost` | 0.6420 | 0.3580 |
| 3 | `lgbm` | 0.6402 | 0.3598 |
| 4 | `xgb_limitdepth` | 0.6371 | 0.3629 |
| 5 | `rf` | 0.5882 | 0.4118 |

> Best validation loss found per estimator type using **r2**.
> FLAML minimises loss internally; Score ≈ 1 − loss. Test-set metrics above are authoritative.
