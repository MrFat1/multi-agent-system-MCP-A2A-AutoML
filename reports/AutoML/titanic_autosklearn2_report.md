# auto-sklearn2 Report — titanic

**Date:** 2026-06-02 22:20
**Time limit:** 300 s &nbsp;|&nbsp; **Random state:** 42

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
| Model | `Unknown` |
| Preprocessor | `minmax_scaler` |
| Models tried | 12 |
| Approx. execution time | ~13.4 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **0.7949** |
| F1 (weighted) | **0.7938** |

---

## Leaderboard (Top 5)

| # | Model | CV Score | Best |
|---|-------|----------|------|
| 1 | `minmax_scaler_random_forest` | 0.7789 |  |
| 2 | `standard_scaler_random_forest` | 0.7773 |  |
| 3 | `robust_scaler_random_forest` | 0.7741 |  |
| 4 | `robust_scaler_bagging` | 0.7564 |  |
| 5 | `minmax_scaler_bagging` | 0.7548 |  |

> Scores are cross-validation scores using **f1_weighted**.
> Test-set metrics above are authoritative.
> auto-sklearn2 uses CV-based model selection (no Bayesian optimisation or ensembles).
