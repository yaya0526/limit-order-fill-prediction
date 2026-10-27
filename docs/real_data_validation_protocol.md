# R5 licensed real-data validation protocol

## Status

The R5 **implementation-readiness package is complete**, but empirical R5 validation has not started. The current workspace contains no licensed event-level exchange data, entitlement evidence, venue rulebook snapshot, or independent audit sign-off. R1–R4 results therefore remain synthetic evidence only.

This protocol defines the gates that must be passed before any real-market performance number is produced. It is designed to prevent a public demonstration project from quietly turning an unverified sample into a live-execution claim.

## Gate 1 — provenance and entitlement

Before data are read by the modelling pipeline, a local manifest must record:

- an explicit dataset identifier and provider;
- a license or entitlement reference approved for the intended analysis;
- venue, symbols, date range, timezone, timestamp unit, and tick size;
- every input file, its role, and its SHA-256 digest;
- confirmation that derived aggregate results may be published.

The machine-readable contract is [`real_data_manifest.schema.json`](../schemas/real_data_manifest.schema.json). Raw files, license documents, account details, and proprietary rulebooks must remain outside the public repository.

## Gate 2 — normalized event stream

The first supported intake format is UTF-8 CSV with one normalized level-3 event per row. Required fields are defined in [`normalized_l3_event.schema.json`](../schemas/normalized_l3_event.schema.json):

| Field | Meaning |
|---|---|
| `dataset_id`, `venue`, `symbol`, `trading_day` | Provenance and stream partition |
| `event_timestamp_ns` | Exchange or provider event time, normalized to nanoseconds |
| `sequence_number` | Strictly increasing sequence within symbol/day |
| `message_type` | `add`, `modify`, `cancel`, or `execute` |
| `side`, `order_id` | Order direction and stable venue/provider identifier |
| `price_ticks`, `quantity` | Integer price grid and positive event quantity |

The intake tool rejects decreasing timestamps, non-increasing sequence numbers, undeclared symbols or dates, digest mismatches, invalid event types, missing execution/cancellation messages, and any precomputed future-label columns.

Run it locally after placing the authorized files outside version control:

```bash
python scripts/check_real_data_readiness.py \
  --manifest /authorized/path/manifest.json \
  --events /authorized/path/events.csv
```

Add `--json` for a machine-readable report. Passing this tool means only that the package is structurally ready for reconstruction; it does not certify exchange semantics or model validity.

## Gate 3 — venue-semantic reconstruction

Before feature generation, write and independently review a venue adapter that specifies:

1. price-time or pro-rata priority rules and any order-type exceptions;
2. how order modifications affect priority;
3. partial-fill accounting and residual quantity;
4. trade-to-resting-order attribution;
5. hidden, iceberg, auction, and off-book event treatment;
6. session states, halts, crossed books, and corporate-action adjustments;
7. timestamp source, feed latency, and clock synchronization assumptions.

Reconstruction must replay each symbol/day from a clean state, preserve event order, never create negative displayed quantity, and reconcile provider control totals. Days that fail reconciliation are quarantined rather than repaired silently.

## Gate 4 — decision-time and label specification

The real-data study must predeclare:

- the simulated order submission rule and observation time $t_0$;
- latency scenarios in physical time, selected before test evaluation;
- whether the target is any fill or complete fill;
- how partial fills, cancellations, amendments, expiry, and market close are censored;
- whether the studied order joins, improves, or sits behind displayed priority;
- 5-, 30-, and 60-second eligibility rules.

Every feature must be reproducible using events observable no later than the latency-adjusted $t_0$. Post-$t_0$ events may be used only for labels and economic outcomes.

## Gate 5 — chronological evaluation

The evaluation plan is frozen before opening the final period:

- split complete trading days into train, validation, and test blocks;
- keep corporate-action and session changes in chronological order;
- select features, model family, calibration, and decision thresholds on train/validation only;
- report ROC-AUC, PR-AUC, Brier score, ECE, action rate, precision, recall, and horizon prevalence;
- measure realized adverse selection and implementation shortfall only when the license and price path support them;
- include day/block uncertainty and market-regime slices;
- compare with the public synthetic benchmark without pooling the datasets.

The final test is opened once. A failed result remains part of the record; it is not converted into a development set.

## Gate 6 — independent audit and publication boundary

An independent reviewer must sign off on:

- provenance and entitlement evidence;
- normalized-field mapping and timestamp alignment;
- venue rules and queue-reconstruction invariants;
- leakage tests and chronological split hashes;
- censoring, partial-fill, amendment, and latency assumptions;
- the exact model/threshold freeze record;
- which aggregate outputs may be published.

Public artifacts should contain code, schemas, configuration hashes, aggregate metrics, and non-sensitive figures only. No proprietary messages, order identifiers, participant information, credentials, license text, or row-level reconstruction output should be committed.

## R5 acceptance criterion

R5 becomes complete only when all six gates pass on properly licensed data and the independent audit is recorded before performance claims are published. Until then, the correct repository status is **ready for authorized data; empirical validation blocked by external data and audit requirements**.
