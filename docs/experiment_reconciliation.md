# Reconciliation of SJTU summer-school portfolio and repository results

Review date: 2026-11-15. Status: **document audit; not an independent model rerun**.

## Sources and boundary

- The 11-page November 2026 `SJTU_Limit_Order_Execution_Report.pdf` supplied by Xiaoya Wu.
- The 19-page `SJTU_Portfolio_Overleaf_Project-2.pdf` (historical presentation).
- Repository `results/final_test_metrics.csv`, `results/selected_model_policy.json`, `results/validation_candidate_summary.csv`, `results/shap_global_importance.csv`, `README.md` and `RESULTS.md`.

## Frozen held-out model results

| Horizon | Frozen model | Feature set | Eligible held-out n | ROC-AUC | PR-AUC | Brier | F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5s | logistic regression | B: order + LOB | 1,460 | 0.616 | 0.170 | 0.102 | 0.158 |
| 30s | random forest | C: order + LOB + flow | 1,311 | 0.650 | 0.600 | 0.230 | 0.611 |
| 60s | random forest | C: order + LOB + flow | 1,234 | 0.690 | 0.805 | 0.205 | 0.789 |

These values agree with the uploaded research report and current repository final-test result files. They are not independent replications.

Validation model selection used PR-AUC after calibration selection by validation Brier score; validation F1 supplied the frozen operating thresholds. The stored thresholds are 0.1891, 0.3472, and 0.3434. The 60s predicted positive *decision* rate is 96.4%, not the mean predicted probability.

## Historical presentation — non-comparable results awaiting provenance

| Original slide | Reported figures | Review status |
| --- | --- | --- |
| 11 | Logistic ROC-AUC: 0.727, 0.712, 0.719 (5/30/60s) | Not traceable to the present frozen test records |
| 12 | TCN/CNN–LSTM 0.717 (5s); CatBoost 0.717 (30s), 0.731 (60s) | Distinct model families, run metadata/split unverified |
| 13 | CNN–LSTM history-window sensitivity; 5 snapshots ROC-AUC 0.7192 | Historical exploratory result, not evidence of robust improvement |
| 16 | CatBoost mean |SHAP|: queue 0.226, normalized queue 0.181, price distance 0.150 | Different model and/or SHAP output unit; cannot directly compare |

The new reviewer-oriented report and slides use the frozen held-out figures, rather than silently combining experiments. Earlier results are **not declared erroneous**: confirm the original split, prediction labels, eligible sample, random seed, preprocessing, fitting code, model versions and metric/SHAP scale before restoring them to a unified comparison.

## SHAP reconciliation

In the repository `results/shap_global_importance.csv`, the measured quantity is `mean_absolute_shap_probability`. At 30s it includes:

- `queue_ahead_qty`: 0.0385444
- `price_distance_ticks`: 0.0296306
- `aggressive_trade_volume_against_order_10s`: 0.0279825
- `normalized_queue_ahead`: 0.0220621

The previous CatBoost values are not equivalent. Any chart must specify model, horizon, data split, SHAP output unit and code provenance.

## Authorship and publication checks

The summer-school work was collaborative. Xiaoya Wu's stated individual focus was feature engineering, identifying potential target leakage, and SHAP-based interpretation. Confirm permissions to distribute shared figures, code, source teaching data, and teammate contributions. The public repository should continue excluding raw training files.

## Next audit step

Reproduce each historical performance table from the original summer-school scripts and matched predictions before making quantitative claims across model families. Maintain synthetic-versus-live-market limitations.

