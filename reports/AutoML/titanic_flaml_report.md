# FLAML Report — titanic

**Date:** 2026-06-02 20:59
**Time limit:** 120 s

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
| Model | `lgbm` |
| Best validation loss | 0.1742 |
| Best iteration | 184 |
| Approx. execution time | ~120.1 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **0.7821** |
| F1 (weighted) | **0.7786** |

---

## Per-Estimator Leaderboard (Top 5)

| # | Model | Score (F1-weighted) | Val. loss |
|---|-------|--------------|-----------|
| 1 | `lgbm` | 0.8258 | 0.1742 |
| 2 | `xgboost` | 0.8116 | 0.1884 |
| 3 | `xgb_limitdepth` | 0.8080 | 0.1920 |
| 4 | `catboost` | 0.7987 | 0.2013 |
| 5 | `extra_tree` | 0.7828 | 0.2172 |

> Best validation loss found per estimator type using **f1_weighted**.
> FLAML minimises loss internally; Score ≈ 1 − loss. Test-set metrics above are authoritative.
