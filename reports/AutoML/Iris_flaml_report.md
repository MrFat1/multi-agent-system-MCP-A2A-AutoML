# FLAML Report — Iris

**Date:** 2026-06-02 20:55
**Time limit:** 120 s

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
| Model | `extra_tree` |
| Best validation loss | 0.0171 |
| Best iteration | 48 |
| Approx. execution time | ~235.6 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **0.9333** |
| F1 (weighted) | **0.9333** |

---

## Per-Estimator Leaderboard (Top 5)

| # | Model | Score (F1-weighted) | Val. loss |
|---|-------|--------------|-----------|
| 1 | `extra_tree` | 0.9829 | 0.0171 |
| 2 | `lgbm` | 0.9659 | 0.0341 |
| 3 | `catboost` | 0.9659 | 0.0341 |
| 4 | `xgboost` | 0.9563 | 0.0437 |
| 5 | `rf` | 0.9392 | 0.0608 |

> Best validation loss found per estimator type using **f1_weighted**.
> FLAML minimises loss internally; Score ≈ 1 − loss. Test-set metrics above are authoritative.
