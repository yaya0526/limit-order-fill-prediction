# Research results

This page is the reviewer-facing summary of the completed analysis. Values below come from the frozen held-out test evaluation; model, calibration, and threshold choices were made on the training and validation periods before the test set was opened.

## Executive summary

- Fill probability is learnable above the horizon-specific prevalence baseline, but signal quality varies sharply by horizon.
- The 5-second problem is the weakest and shows the largest validation-to-test deterioration.
- Random forests using order, LOB, and recent-flow features are selected at 30 and 60 seconds.
- The 50-snapshot history block does not produce a robust, repeatable gain over the simpler feature sets.
- Calibration error is numerically small, but classification thresholds can still create an impractical operating policy—most clearly at 60 seconds.
- Queue-ahead quantity, price distance, aggressive volume against the order, and normalized queue position are leading model drivers.

## Frozen held-out test results

| Horizon | Frozen policy | Calibration | Threshold | Eligible | Fill rate | Predicted positive rate | ROC-AUC | PR-AUC | Brier | ECE | Accuracy | Precision | Recall | F1 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5s | Logistic, order + LOB | Sigmoid | 0.189 | 1,460 | 0.118 | 0.064 | 0.616 | 0.170 | 0.102 | 0.014 | 0.847 | 0.223 | 0.122 | 0.158 |
| 30s | Random forest, order + LOB + flow | Sigmoid | 0.347 | 1,311 | 0.445 | 0.642 | 0.650 | 0.600 | 0.230 | 0.035 | 0.577 | 0.517 | 0.746 | 0.611 |
| 60s | Random forest, order + LOB + flow | Sigmoid | 0.343 | 1,234 | 0.643 | 0.964 | 0.690 | 0.805 | 0.205 | 0.028 | 0.660 | 0.658 | 0.985 | 0.789 |

PR-AUC exceeds the observed fill rate at each horizon, which indicates ranking signal beyond a prevalence-only predictor. This does not mean every threshold is useful. At 60 seconds the policy marks 96.4% of eligible orders as positive, producing only 33 true negatives against 407 false positives. The high F1 is therefore driven partly by the high base rate and near-universal positive prediction.

Source: [`results/final_test_metrics.csv`](results/final_test_metrics.csv) and [`results/final_test_summary.json`](results/final_test_summary.json).

## Generalization from validation to test

| Horizon | PR-AUC change | F1 change | Interpretation |
|---:|---:|---:|---|
| 5s | -0.077 | -0.164 | Material degradation; short-horizon conclusions should be conservative |
| 30s | -0.005 | +0.012 | Broadly stable |
| 60s | +0.008 | +0.022 | Broadly stable, but threshold selectivity remains weak |

The 5-second gap is the clearest sign that the development-period estimate was optimistic. See [`results/validation_test_comparison.csv`](results/validation_test_comparison.csv) and the figure below.

![Validation versus test metrics](reports/figures/final_test_validation_comparison.png)

## Rolling-origin robustness

The frozen v0.1.0 feature sets, model families, and hyperparameters were refitted across three expanding-origin windows. Each window used only past data for sigmoid calibration and an immediately preceding two-day validation period for its F1 threshold. The six rolling test days do not overlap, and the original three-day final test period was not reopened.

| Horizon | PR-AUC, mean ± SD | Brier, mean ± SD | F1, mean ± SD | Pooled predicted positive rate |
|---:|---:|---:|---:|---:|
| 5s | 0.208 ± 0.021 | 0.093 ± 0.005 | 0.241 ± 0.027 | 0.433 |
| 30s | 0.616 ± 0.002 | 0.221 ± 0.004 | 0.617 ± 0.016 | 0.734 |
| 60s | 0.792 ± 0.004 | 0.212 ± 0.002 | 0.779 ± 0.017 | 0.941 |

The 30- and 60-second policies retain stable ranking performance, while the 5-second task remains weaker and more threshold-sensitive. At 60 seconds the predicted positive rate stays between 92.4% and 96.0% in all three windows, confirming that weak selectivity is persistent rather than isolated to the final holdout.

![Rolling-origin robustness](reports/figures/rolling_origin_robustness.png)

The full design, trading-day block intervals, and post-selection limitation are documented in [`docs/rolling_origin_robustness.md`](docs/rolling_origin_robustness.md). Machine-readable results are in [`results/rolling_origin_window_metrics.csv`](results/rolling_origin_window_metrics.csv) and [`results/rolling_origin_summary.json`](results/rolling_origin_summary.json).

## Feature-set conclusions

The experiment compares cumulative feature groups:

- **A — order state:** side, size, price distance, queue position, and order attributes;
- **B — order + LOB:** contemporaneous depth, spread, and imbalance;
- **C — order + LOB + flow:** recent trades, cancellations, and market activity;
- **D — order + LOB + flow + history:** engineered statistics from 50 pre-submission snapshots.

The history block is a useful negative result. In the fixed 30-second random-forest ablation, D reaches PR-AUC 0.598 versus 0.593 for C, a small gain. After candidate tuning, D reaches 0.601 versus 0.604 for C. Because the sign and size of the increment are not robust, no final frozen policy uses D.

The selected 30- and 60-second policies use C, while the selected 5-second policy uses B. Detailed development results are in [`results/validation_model_feature_results.csv`](results/validation_model_feature_results.csv).

## Explainability findings

Global explanations repeatedly elevate:

1. `queue_ahead_qty`;
2. `price_distance_ticks`;
3. `aggressive_trade_volume_against_order_10s`;
4. `normalized_queue_ahead`.

These variables are economically plausible: orders farther from the touch or behind more queued volume should be harder to fill, while aggressive opposite-side flow can consume standing liquidity. The explanations remain associational and model-specific; they are not causal estimates.

![Global SHAP feature importance](reports/figures/brief_shap_global_importance.png)

## Slice and robustness findings

Side-level performance is broadly similar, while variation across market-activity regimes and individual test days is larger. These slices are descriptive because some subgroups are small. They should guide follow-up diagnostics rather than be treated as independently powered hypothesis tests.

See [`results/final_test_slice_metrics.csv`](results/final_test_slice_metrics.csv) for the full subgroup table.

## Artifact index

| Artifact | Purpose |
|---|---|
| [`results/final_test_metrics.csv`](results/final_test_metrics.csv) | Main held-out metric table |
| [`results/final_test_summary.json`](results/final_test_summary.json) | Frozen policies, thresholds, and final summary |
| [`results/validation_test_comparison.csv`](results/validation_test_comparison.csv) | Development-to-test changes |
| [`results/validation_model_feature_results.csv`](results/validation_model_feature_results.csv) | Model and feature-set selection results |
| [`results/final_test_slice_metrics.csv`](results/final_test_slice_metrics.csv) | Side, regime, and day slices |
| [`results/rolling_origin_window_metrics.csv`](results/rolling_origin_window_metrics.csv) | R1 metrics for three non-overlapping test windows |
| [`results/rolling_origin_summary.json`](results/rolling_origin_summary.json) | R1 aggregates, day-block intervals, and acceptance record |
| [`results/brief_completion_summary.json`](results/brief_completion_summary.json) | Checklist against the project brief |
| [`reports/figures/`](reports/figures/) | Calibration, discrimination, ablation, and explanation plots |
| [`reports/`](reports/) | Technical report and presentation |

## Interpretation boundaries

- **Synthetic data:** the results describe the supplied generator, not live AAPL execution.
- **Single symbol:** cross-sectional transfer is untested.
- **Fixed horizons:** this is not a full time-to-event or competing-risk model.
- **Threshold dependence:** ranking and calibration metrics do not guarantee a useful execution decision rule.
- **Uncertainty:** the final holdout still uses row-level intervals; the R1 extension adds day-block intervals but has only six independent rolling test days.
- **No economic backtest:** fill classification is not equivalent to realized implementation shortfall or trading profitability.

The full evidence boundary, follow-up questions, and measurable acceptance criteria are documented in [`docs/limitations_and_roadmap.md`](docs/limitations_and_roadmap.md).

## Reproducibility and provenance

The study uses the supplied chronological 14/3/3-day split and random seed 42. Final policies were frozen before the one-time test evaluation. R1 then audits those specifications across three earlier rolling origins and records its post-selection limitation explicitly. Raw teaching data are not redistributed, but derived outputs and committed notebook results are included for auditability. See [`data/README.md`](data/README.md) for the local input contract.
