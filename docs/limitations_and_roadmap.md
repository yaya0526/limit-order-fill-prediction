# Limitations and research roadmap

This document separates what the repository currently demonstrates from what a production-grade execution study would still need to establish. It is intentionally stricter than a feature wish list: every proposed extension has a concrete question and an acceptance criterion.

## Current evidence boundary

The repository demonstrates a disciplined modelling workflow on deterministic synthetic teaching data. It shows how to construct fixed-horizon fill labels, preserve chronology, compare cumulative feature groups, calibrate probabilities, freeze a validation-selected policy, and evaluate it once on a held-out period.

It does **not** demonstrate live AAPL execution quality, a profitable trading strategy, causal market mechanisms, or portability to another symbol, venue, or period.

## Main limitations

### 1. Synthetic, single-symbol data

The supplied data emulate AAPL-like order-book behaviour but do not reproduce every exchange rule, latency effect, hidden-liquidity interaction, or market regime. Because only one synthetic symbol and one 20-day sequence are available, cross-sectional and long-run temporal transfer are untested.

**Consequence:** the measured discrimination and calibration describe this generator and split, not real Nasdaq execution.

### 2. Limited chronological coverage

The 14/3/3-day split protects time order and keeps the test period untouched during model selection. R1 adds three non-overlapping rolling test windows and day-block uncertainty, but all windows still come from one 20-day synthetic sequence. The 5-second variation shows why short-horizon conclusions remain sensitive to period and threshold choice.

**Consequence:** short-run temporal dispersion is measured more clearly, but long-run regime transfer remains untested.

### 3. Fixed-horizon binary targets

Separate 5-, 30-, and 60-second labels are easy to interpret, but they discard the exact time-to-fill and treat each horizon as a different classification problem. Eligibility masks handle end-of-sample right-censoring; they do not model censoring directly.

**Consequence:** the three probabilities are not guaranteed to be mutually coherent, and the workflow cannot estimate a full fill-time distribution.

### 4. Probability quality is not decision utility

ROC-AUC, PR-AUC, Brier score, and calibration error evaluate predictions, not the economic value of an order-placement decision. The 60-second policy predicts 96.4% of eligible orders as fills against a 64.3% observed rate. Its high F1 therefore coexists with weak selectivity.

**Consequence:** a statistically reasonable model can still produce an operationally poor threshold.

### 5. Queue and execution realism

Queue position, order flow, and book state are informative in the supplied data, but the repository does not reconstruct exchange messages or explicitly simulate latency, partial fills, amendments, hidden liquidity, or venue priority rules.

**Consequence:** feature importance should not be interpreted as a complete structural explanation of fill mechanics.

### 6. Uncertainty and dependence

The reported confidence intervals use row-level bootstrap resampling. Orders from the same day or market regime can be dependent, so row-level resampling may understate uncertainty.

**Consequence:** the intervals are descriptive and should not be used as formal guarantees of out-of-sample performance.

### 7. Interpretability is associational

SHAP, LIME, permutation importance, and response profiles explain how fitted models use features. Correlated predictors and model extrapolation can change individual attributions.

**Consequence:** explanation outputs support auditing and hypothesis generation, not causal claims.

## Prioritized research roadmap

### R1 — Rolling-origin robustness — complete

**Question:** do model rankings and calibration persist across multiple chronological test windows?

**Method:** repeat training and evaluation with several expanding or rolling windows; report per-window and aggregated metrics. Add day- or block-level bootstrap intervals.

**Acceptance criterion:** every selected policy has results from at least three non-overlapping test windows, with dispersion reported alongside mean performance.

**Completion:** three expanding-origin windows now report per-window metrics, mean, standard deviation, range, and pooled trading-day block intervals. The original final test period remains untouched. See [`rolling_origin_robustness.md`](rolling_origin_robustness.md).

### R2 — Time-to-fill modelling

**Question:** can a discrete-time hazard or survival model produce coherent probabilities across horizons while handling censoring directly?

**Method:** estimate a fill-time survival curve and compare its 5-, 30-, and 60-second cumulative probabilities with the current classifiers.

**Acceptance criterion:** horizon probabilities are monotone by construction and are evaluated with time-dependent discrimination and calibration metrics.

### R3 — Decision-aware thresholds

**Question:** which threshold is useful when missed fills, false fills, adverse selection, and waiting time have different costs?

**Method:** define a transparent utility or cost matrix, choose thresholds on validation data, and report sensitivity across plausible cost assumptions.

**Acceptance criterion:** the operating point is justified by a stated objective and remains selective on held-out data; F1 is no longer the sole threshold criterion.

### R4 — Independent market simulation

**Question:** are conclusions robust to a different data-generating process?

**Method:** test on an independently configured simulator with changed spread, depth, volatility, and order-flow regimes.

**Acceptance criterion:** the evaluation documents both successful transfer and failure regimes without retuning on the final test scenario.

### R5 — Real-data validation

**Question:** which findings survive on licensed event-level exchange data?

**Method:** reconstruct queue state with explicit venue rules, latency assumptions, partial fills, and order amendments. Keep the current synthetic workflow as a public reproducible benchmark.

**Acceptance criterion:** provenance, entitlements, timestamp alignment, leakage checks, and execution assumptions receive an independent audit before performance claims are made.

## Recommended execution order

1. R1 is complete; retain it as the temporal-stability baseline.
2. Use R2 to replace disconnected horizon classifiers with a coherent event-time formulation.
3. Add R3 before describing any result as operationally useful.
4. Use R4 as a public robustness benchmark.
5. Attempt R5 only with properly licensed data and documented exchange semantics.

The next research milestone is **coherence before another classifier**: use R2 to model time to fill directly, then use R3 to connect thresholds to an explicit decision objective.
