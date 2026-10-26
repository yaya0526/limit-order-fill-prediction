# Coherent time-to-fill model

This extension completes roadmap item R2 by replacing three disconnected horizon classifiers with a discrete-time fill-hazard model. It estimates conditional fill hazards over `(0, 5]`, `(5, 30]`, and `(30, 60]` seconds, then converts them into a cumulative fill distribution.

## Why a hazard model

Independent classifiers can produce contradictory horizon probabilities, such as a lower estimated chance of filling by 60 seconds than by 30 seconds. A hazard construction avoids that failure by combining conditional interval probabilities:

```text
P(fill by 5s)  = h1
P(fill by 30s) = 1 - (1 - h1)(1 - h2)
P(fill by 60s) = 1 - (1 - h1)(1 - h2)(1 - h3)
```

Because every interval hazard lies between zero and one, the cumulative probabilities cannot decrease as the horizon grows.

## Censoring and risk sets

The source defines `observed_duration_s` as the minimum of first-fill time and cancellation/expiry time. The model uses that duration directly:

- an order contributes an event if it fills inside the interval;
- an order contributes a non-event only when it is still observed through the interval end; and
- an order cancelled or expired inside the interval is excluded from that interval rather than labelled as a non-fill.

| Interval | Training risk rows | Fills in interval | Conditional event rate |
|---:|---:|---:|---:|
| (0, 5] seconds | 6,807 | 711 | 10.45% |
| (5, 30] seconds | 5,462 | 1,870 | 34.24% |
| (30, 60] seconds | 3,196 | 1,076 | 33.67% |

Each interval uses logistic regression with the order, LOB, and recent-flow feature set C. Forward-only out-of-fold predictions from the 14 training days fit a sigmoid calibrator for each interval. The design uses fixed regularization (`C=1.0`) and does not search model families on the final test period.

## Time-dependent results

The metrics below evaluate cumulative fill probabilities among orders whose outcome is observable at each horizon. The independent-classifier values are the previously committed frozen test results.

| Horizon | Hazard PR-AUC | Independent PR-AUC | Hazard Brier | Independent Brier | Hazard ECE |
|---:|---:|---:|---:|---:|---:|
| 5s | 0.162 | 0.170 | 0.103 | 0.102 | 0.018 |
| 30s | 0.614 | 0.600 | 0.226 | 0.230 | 0.034 |
| 60s | 0.815 | 0.805 | 0.203 | 0.205 | 0.043 |

![Coherent time-to-fill comparison](../reports/figures/time_to_fill_coherence.png)

### Interpretation

- The coherent model has zero monotonicity violations across all 1,500 validation and 1,500 test orders.
- At 30 and 60 seconds it slightly improves both ranking and Brier score relative to the separate frozen classifiers.
- At 5 seconds it is slightly worse. A common feature set and event-time structure improve coherence but do not solve the sparse short-horizon signal problem.
- The cumulative model addresses probability coherence, not decision utility. Its threshold-dependent F1 values use a descriptive 0.5 cutoff and should not replace the R3 cost-based policy analysis.

## Acceptance decision

R2 passes its stated acceptance criterion:

- horizon probabilities are monotone by construction;
- validation monotonicity violations: 0 of 1,500 rows;
- test monotonicity violations: 0 of 1,500 rows; and
- horizon-specific ROC-AUC, PR-AUC, Brier score, and ECE are reported for validation and test data.

## Limits

The three intervals provide a coarse event-time distribution rather than a continuous-time curve. Cancellation and expiry are treated as independent right-censoring; a competing-risk model would be needed if cancellation itself were a target outcome. This is a post-hoc extension on the same deterministic synthetic sequence, not independent market validation.

## Reproduction

The model needs only the Excel workbook:

```bash
python scripts/run_time_to_fill.py --excel data/raw/AI_Order_Execution.xlsx
```

## Artifacts

| Artifact | Purpose |
|---|---|
| [`results/time_to_fill_metrics.csv`](../results/time_to_fill_metrics.csv) | Validation/test discrimination, calibration, and classifier comparison |
| [`results/time_to_fill_risk_set_audit.csv`](../results/time_to_fill_risk_set_audit.csv) | Censoring-aware risk-set counts and event rates |
| [`results/time_to_fill_summary.json`](../results/time_to_fill_summary.json) | Structured model design and R2 acceptance record |
| [`scripts/run_time_to_fill.py`](../scripts/run_time_to_fill.py) | Reproducible experiment entry point |
