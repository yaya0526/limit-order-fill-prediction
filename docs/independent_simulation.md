# R4 independent simulation stress test

## Research question

Do the frozen 5-, 30-, and 60-second models retain useful ranking or calibration when spread, displayed depth, volatility, order flow, and queue mechanics come from a different data-generating process?

This is a transfer stress test, not an attempt to reproduce Nasdaq. The simulator is deliberately small, deterministic, and inspectable. Its parameters are code constants in [`src/market_simulation.py`](../src/market_simulation.py); it does not read or estimate parameters from the source workbook.

## Design

Four scenarios generate 4,000 orders each with seed `20261026`:

| Scenario | Main shift | Fill mechanism |
|---|---|---|
| `liquid_consistent` | Tight spread, deeper displayed depth, faster flow | Visible queue depletion |
| `wide_thin_volatile` | Wide spread, thin depth, high volatility, slower flow | Visible queue depletion |
| `flow_reversal` | High displayed aggressive flow induces replenishment | Flow signal is partly reversed |
| `hidden_queue` | Low displayed depth but unobserved priority ahead | Hidden queue dominates fills |

For visible-queue scenarios, simulated fill time is the sum of a distance-to-touch delay and a stochastic queue-depletion time:

$$
T_{fill} = \left(T_{touch} + \frac{Q_{ahead} + 0.5Q_{remaining}}{r_{trade}+r_{cancel}+1}\right)\epsilon,
$$

where $\epsilon$ is fixed-seed log-normal noise. The replenishment and hidden-queue scenarios alter the numerator and effective flow rate explicitly. Cumulative labels are then $1[T_{fill}\le h]$ for $h\in\{5,30,60\}$, so they are monotone by construction.

The source training period is used only to reconstruct the already frozen estimator and sigmoid calibrator. The original validation/test rows are not scored in R4. Thresholds come unchanged from the R3 precision-first policy; there is no scenario-specific fitting, calibration, or threshold selection.

## Predeclared transfer rule

PR-AUC is compared with each scenario's own fill-rate baseline. A horizon has ranking transfer when:

$$
\text{PR-AUC} - \text{fill rate} > 0.02.
$$

Positive Brier skill against the constant-prevalence forecast additionally indicates probability-calibration transfer. If a scenario has ranking transfer at two or more horizons, it is a successful-transfer regime; if it transfers at none, it is a failure regime. All-zero target slices are failures because discrimination is not identifiable.

## Results

| Scenario | Horizon | Fill rate | PR-AUC | PR lift | Brier skill | Frozen-policy positive rate | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Liquid consistent | 5s | 0.003 | 0.218 | 0.215 | -1.855 | 0.000 | Ranking only |
| Liquid consistent | 30s | 0.155 | 0.620 | 0.464 | -0.063 | 0.000 | Ranking only |
| Liquid consistent | 60s | 0.434 | 0.797 | 0.363 | 0.151 | 0.028 | Ranking + calibration |
| Wide/thin/volatile | 5s | 0.001 | 0.673 | 0.672 | -2.160 | 0.000 | Ranking only |
| Wide/thin/volatile | 30s | 0.104 | 0.596 | 0.493 | -0.034 | 0.000 | Ranking only |
| Wide/thin/volatile | 60s | 0.359 | 0.742 | 0.383 | 0.173 | 0.010 | Ranking + calibration |
| Flow reversal | 5s | 0.000 | — | — | — | 0.000 | Failure |
| Flow reversal | 30s | 0.019 | 0.156 | 0.137 | -5.565 | 0.001 | Ranking only |
| Flow reversal | 60s | 0.139 | 0.371 | 0.232 | -1.250 | 0.042 | Ranking only |
| Hidden queue | 5s | 0.000 | — | — | — | 0.000 | Failure |
| Hidden queue | 30s | 0.000 | — | — | — | 0.001 | Failure |
| Hidden queue | 60s | 0.006 | 0.021 | 0.015 | -47.216 | 0.031 | Failure |

![Independent simulation transfer](../reports/figures/independent_simulation_transfer.png)

Three scenarios retain ranking information at least two horizons, but calibration transfers only at 60 seconds in the two visible-queue regimes. The hidden-queue mechanism fails at all horizons. Frozen cost-aware thresholds also become almost inactive in the shifted scenarios, showing that a portable ranker is not automatically a portable decision policy.

The result is intentionally mixed: feature relationships can survive moderate regime changes, while base rates, probability calibration, and action thresholds do not transfer reliably.

## Acceptance and limitations

R4 passes its stated repository acceptance criterion because it records both successful transfer and a complete failure regime without retuning on any scenario. This is limited synthetic evidence only.

The generator does not model an exchange message stream, venue-specific price-time priority, latency, amendments, partial-fill paths, or realistic hidden-liquidity behavior. Its failure modes are controlled counterexamples, not estimates of their frequency in live markets. R5 therefore remains necessary before any real-execution claim.

Machine-readable results are in [`independent_simulation_metrics.csv`](../results/independent_simulation_metrics.csv) and [`independent_simulation_summary.json`](../results/independent_simulation_summary.json).
