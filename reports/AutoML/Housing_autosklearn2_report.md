# auto-sklearn2 Report — Housing

**Date:** 2026-06-02 22:21
**Time limit:** 300 s &nbsp;|&nbsp; **Random state:** 42

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
| Model | `Unknown` |
| Preprocessor | `minmax_scaler` |
| Models tried | 66 |
| Approx. execution time | ~18.3 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| R² | **0.6706** |
| RMSE | **1290368.3772** |

---

## Leaderboard (Top 5)

| # | Model | CV Score | Best |
|---|-------|----------|------|
| 1 | `minmax_scaler_poisson` | 0.6630 |  |
| 2 | `robust_scaler_poisson` | 0.6630 |  |
| 3 | `standard_scaler_poisson` | 0.6630 |  |
| 4 | `minmax_scaler_kernel_ridge` | 0.6506 |  |
| 5 | `robust_scaler_sgd` | 0.6505 |  |

> Scores are cross-validation scores using **r2**.
> Test-set metrics above are authoritative.
> auto-sklearn2 uses CV-based model selection (no Bayesian optimisation or ensembles).
