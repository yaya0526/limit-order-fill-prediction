# Limit Order Fill Probability Prediction

## Overview

This project studies whether a resting limit order will be filled within
5, 30, or 60 seconds using order-level, queue, limit-order-book, and
recent order-flow information.

## Research Question

Given the information available at observation time t0, what is the
probability that an active limit order will be filled within a fixed horizon?

## Dataset

The project uses deterministic synthetic teaching data for a single symbol,
AAPL. It contains 10,000 order-level observations and 50 historical
order-book snapshots per order.

The data are synthetic and should not be interpreted as evidence about
real AAPL or Nasdaq execution.

## Prediction Targets

- Fill within 5 seconds
- Fill within 30 seconds
- Fill within 60 seconds

Only observations eligible at the corresponding horizon are included.

## Data Splitting

The project uses the supplied chronological split:

- Training: first 14 trading days
- Validation: next 3 trading days
- Test: final 3 trading days

The test set is reserved for final evaluation.

## Repository Structure

- `data/`: data instructions and local data files
- `docs/`: feature definitions and project notes
- `notebooks/`: exploratory and modelling notebooks
- `src/`: reusable project code
- `tests/`: validation tests
- `reports/`: final figures and report
- `results/`: model evaluation results

## Feature Reference

See [`docs/feature_reference.md`](docs/feature_reference.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
