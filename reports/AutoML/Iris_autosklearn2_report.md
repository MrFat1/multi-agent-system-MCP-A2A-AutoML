# auto-sklearn2 Report — Iris

**Date:** 2026-06-02 22:19
**Time limit:** 300 s &nbsp;|&nbsp; **Random state:** 42

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
| Model | `Unknown` |
| Preprocessor | `standard_scaler` |
| Models tried | 49 |
| Approx. execution time | ~14.9 s |

---

## Test-Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **0.9667** |
| F1 (weighted) | **0.9667** |

---

## Leaderboard (Top 5)

| # | Model | CV Score | Best |
|---|-------|----------|------|
| 1 | `standard_scaler_lda` | 0.9830 |  |
| 2 | `minmax_scaler_lda` | 0.9830 |  |
| 3 | `robust_scaler_lda` | 0.9830 |  |
| 4 | `standard_scaler_svc` | 0.9746 |  |
| 5 | `standard_scaler_qda` | 0.9743 |  |

> Scores are cross-validation scores using **f1_weighted**.
> Test-set metrics above are authoritative.
> auto-sklearn2 uses CV-based model selection (no Bayesian optimisation or ensembles).
