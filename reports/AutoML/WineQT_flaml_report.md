# FLAML Report — WineQT

**Date:** 2026-06-02 21:22
**Time limit:** 1000 s

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
| Model | `rf` |
| Best validation loss | 0.3580 |
| Best iteration | 602 |
| Approx. execution time | ~1000.1 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **0.5294** |
| F1 (weighted) | **0.5030** |

---

## Per-Estimator Leaderboard (Top 5)

| # | Model | Score (F1-weighted) | Val. loss |
|---|-------|--------------|-----------|
| 1 | `rf` | 0.6420 | 0.3580 |
| 2 | `extra_tree` | 0.6353 | 0.3647 |
| 3 | `catboost` | 0.6299 | 0.3701 |
| 4 | `xgb_limitdepth` | 0.6285 | 0.3715 |
| 5 | `xgboost` | 0.6248 | 0.3752 |

> Best validation loss found per estimator type using **f1_weighted**.
> FLAML minimises loss internally; Score ≈ 1 − loss. Test-set metrics above are authoritative.
