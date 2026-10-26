# Rolling-origin robustness

This extension tests whether the frozen v0.1.0 model policies behave consistently across several chronological periods. It addresses roadmap item R1 without reopening the original three-day final test set.

## Research question

Do the selected feature sets and model families retain useful discrimination and calibration when they are repeatedly refitted at later origins?

The audit fixes each horizon's v0.1.0 feature set, model family, and hyperparameters. Within each rolling window, it:

1. fits a forward-only sigmoid calibrator from out-of-fold training predictions;
2. refits the frozen model specification on all available training days;
3. selects an F1-maximizing threshold on the next two validation days; and
4. evaluates once on the following two test days.

No rolling test label changes the model specification, calibrator, or threshold used for that test window.

## Window design

| Window | Expanding train period | Validation period | Test period |
|---|---|---|---|
| W1 | 2022-09-06 to 2022-09-15 (8 days) | 2022-09-16 to 2022-09-19 | 2022-09-20 to 2022-09-21 |
| W2 | 2022-09-06 to 2022-09-19 (10 days) | 2022-09-20 to 2022-09-21 | 2022-09-22 to 2022-09-23 |
| W3 | 2022-09-06 to 2022-09-21 (12 days) | 2022-09-22 to 2022-09-23 | 2022-09-26 to 2022-09-27 |

The six rolling test days are non-overlapping. The original final test dates, 2022-09-29 through 2022-10-03, remain untouched.

## Results

The table reports the unweighted mean and sample standard deviation across the three test windows. Pooled positive rates use all eligible rolling-test predictions while preserving each window's own validation-selected threshold.

| Horizon | PR-AUC, mean ± SD | Brier, mean ± SD | F1, mean ± SD | Pooled predicted positive rate |
|---:|---:|---:|---:|---:|
| 5s | 0.208 ± 0.021 | 0.093 ± 0.005 | 0.241 ± 0.027 | 0.433 |
| 30s | 0.616 ± 0.002 | 0.221 ± 0.004 | 0.617 ± 0.016 | 0.734 |
| 60s | 0.792 ± 0.004 | 0.212 ± 0.002 | 0.779 ± 0.017 | 0.941 |

![Rolling-origin metric stability](../reports/figures/rolling_origin_robustness.png)

### Interpretation

- The 30- and 60-second policies retain stable ranking performance. Their cross-window PR-AUC standard deviations are 0.002 and 0.004.
- The 5-second task remains the least reliable. Its PR-AUC ranges from 0.187 to 0.229, and its F1 threshold produces materially different positive rates across windows.
- Brier scores are comparatively stable at every horizon, but probability quality does not make the operating threshold useful by itself.
- The 60-second policy predicts 92.4% to 96.0% of eligible orders as fills in every window. The weak-selectivity concern from the original held-out evaluation therefore persists across time.

Trading-day block bootstrap intervals are included in the machine-readable summary. For example, pooled PR-AUC has 95% block intervals of 0.172–0.229 at 5 seconds, 0.573–0.649 at 30 seconds, and 0.769–0.805 at 60 seconds. These intervals resample whole days to retain within-day dependence.

## Acceptance decision

R1 passes its stated acceptance criterion:

- three non-overlapping chronological test windows are reported for every selected policy;
- mean, standard deviation, minimum, and maximum are reported by metric; and
- day-block bootstrap intervals accompany pooled metrics.

## Limits

This is a post-selection robustness audit, not a fully nested model-selection study. The v0.1.0 hyperparameters were fixed before this extension, but they were originally selected using the broader development period. The audit therefore tests temporal stability of the chosen specifications; it does not provide an unbiased comparison against every discarded specification.

Only six synthetic rolling-test days are available. Day-block intervals are more dependence-aware than row bootstrap intervals, but remain descriptive with so few independent blocks. The study still provides no evidence about live AAPL orders, another symbol, or another simulator.

## Reproduction

With the source workbook available locally:

```bash
python scripts/run_rolling_origin.py --excel data/raw/AI_Order_Execution.xlsx
```

The command writes the window metrics, pooled summary, audit predictions, JSON metadata, and figure. The historical JSON file is not required because the selected policies use only feature sets B and C.

## Artifacts

| Artifact | Purpose |
|---|---|
| [`results/rolling_origin_window_metrics.csv`](../results/rolling_origin_window_metrics.csv) | One row per horizon and rolling window |
| [`results/rolling_origin_summary.csv`](../results/rolling_origin_summary.csv) | Means, dispersion, pooled metrics, and block intervals |
| [`results/rolling_origin_summary.json`](../results/rolling_origin_summary.json) | Design metadata and structured acceptance record |
| [`results/rolling_origin_predictions.csv`](../results/rolling_origin_predictions.csv) | Row-level audit predictions for the six rolling test days |
| [`scripts/run_rolling_origin.py`](../scripts/run_rolling_origin.py) | Reproducible experiment entry point |
