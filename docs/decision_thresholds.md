# Decision-aware thresholds

This extension completes roadmap item R3 by replacing F1-only operating points with validation-selected thresholds tied to an explicit error-cost objective.

## Decision objective

The model predicts whether a resting order will fill within each horizon. Two errors have different operational interpretations:

- **False positive:** predict a fill, but the order does not fill within the horizon. This is a proxy for waiting and opportunity cost.
- **False negative:** predict no fill, but the order would have filled. This is a proxy for missed passive execution.

The costs are normalized sensitivity weights, not currency P&L:

| Scenario | False-positive cost | False-negative cost | Interpretation |
|---|---:|---:|---|
| Precision first (primary) | 3 | 1 | Waiting/opportunity cost dominates |
| Balanced | 1 | 1 | Equal error costs |
| Fill seeking | 1 | 3 | Missed fills dominate |

For each scenario and horizon, the threshold minimizes cost per eligible validation order. Candidate thresholds must select between 1% and 80% of validation orders, preventing both a no-action solution and a near-universal positive policy. Ties are resolved toward the calibrated-probability Bayes threshold. The selected threshold is then carried unchanged to the test period.

## Primary-policy comparison

The table scores both the original F1 threshold and the precision-first threshold under the primary 3:1 cost matrix.

| Horizon | F1 threshold | Cost threshold | F1-policy positive rate | Cost-policy positive rate | Cost-policy precision | Cost-policy recall | F1-policy cost/order | Cost-policy cost/order |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5s | 0.189 | 0.285 | 6.4% | 0.8% | 0.250 | 0.017 | 0.253 | 0.134 |
| 30s | 0.347 | 0.763 | 64.2% | 4.7% | 0.758 | 0.081 | 1.044 | 0.443 |
| 60s | 0.343 | 0.780 | 96.4% | 20.4% | 0.885 | 0.281 | 0.999 | 0.533 |

![Decision-threshold tradeoffs](../reports/figures/decision_threshold_tradeoffs.png)

The precision-first objective sharply reduces normalized cost and removes the near-universal 60-second action rate. It also sacrifices recall. At 5 seconds it acts on only 0.8% of test orders, showing that a high false-positive penalty and a weak short-horizon signal produce an extremely conservative policy.

## Sensitivity

Held-out predicted positive rates change materially with the cost matrix:

| Scenario | 5s | 30s | 60s |
|---|---:|---:|---:|
| Precision first | 0.8% | 4.7% | 20.4% |
| Balanced | 1.9% | 12.5% | 73.2% |
| Fill seeking | 5.8% | 75.1% | 81.8% |

This spread is the central R3 conclusion: a probability model does not determine an execution policy until the costs of the two error types are stated.

## Acceptance decision

R3 passes its stated acceptance criterion:

- thresholds are selected only on validation data;
- F1 is reported but is not the selection objective;
- the primary policy takes at least one positive action at every held-out horizon;
- every primary held-out positive rate stays below the predeclared 80% ceiling; and
- cost sensitivity is reported rather than hiding dependence on one assumption.

## Limits

The workbook does not contain post-decision price paths, spreads paid after a missed fill, or realized implementation shortfall. It therefore cannot estimate adverse selection or waiting cost in currency. The normalized 3:1, 1:1, and 1:3 ratios are transparent policy scenarios, not empirical economic estimates. A production decision study must replace them with measured costs and capacity constraints.

## Reproduction

```bash
python scripts/run_decision_thresholds.py --excel data/raw/AI_Order_Execution.xlsx
```

The command reconstructs the committed test probabilities byte-for-byte, selects thresholds from validation data, and writes only aggregate decision results. Order-level R3 predictions are not committed.

## Artifacts

| Artifact | Purpose |
|---|---|
| [`results/decision_threshold_results.csv`](../results/decision_threshold_results.csv) | Validation/test results for all cost scenarios |
| [`results/decision_threshold_comparison.csv`](../results/decision_threshold_comparison.csv) | Primary cost policy versus original F1 policy |
| [`results/decision_threshold_summary.json`](../results/decision_threshold_summary.json) | Objective, guardrails, acceptance record, and reproduction audit |
| [`scripts/run_decision_thresholds.py`](../scripts/run_decision_thresholds.py) | Reproducible validation-only threshold workflow |
