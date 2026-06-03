# auto-sklearn2 Report — WineQT

**Date:** 2026-06-02 22:18
**Time limit:** 300 s &nbsp;|&nbsp; **Random state:** 42

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
| Model | `Unknown` |
| Preprocessor | `minmax_scaler` |
| Models tried | 49 |
| Approx. execution time | ~22.6 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **0.5441** |
| F1 (weighted) | **0.5108** |

---

## Leaderboard (Top 5)

| # | Model | CV Score | Best |
|---|-------|----------|------|
| 1 | `minmax_scaler_mlp` | 0.6179 |  |
| 2 | `standard_scaler_logistic_regression` | 0.6143 |  |
| 3 | `robust_scaler_logistic_regression` | 0.6131 |  |
| 4 | `minmax_scaler_random_forest` | 0.6081 |  |
| 5 | `standard_scaler_bagging` | 0.6069 |  |

> Scores are cross-validation scores using **f1_weighted**.
> Test-set metrics above are authoritative.
> auto-sklearn2 uses CV-based model selection (no Bayesian optimisation or ensembles).
