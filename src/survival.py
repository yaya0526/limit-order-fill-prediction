"""Discrete-time fill-hazard utilities with explicit censoring risk sets."""

from __future__ import annotations

import numpy as np


HAZARD_INTERVALS = ((0, 5), (5, 30), (30, 60))


def interval_risk_set(frame, start_seconds, end_seconds):
    """Return the observable conditional-risk mask and interval event target.

    Orders filled at or before the interval start have already left the risk set.
    Orders censored inside the interval are excluded rather than labelled as
    non-events. A fill on the interval end belongs to that interval.
    """
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("Hazard intervals must have 0 <= start < end.")

    observed_duration = frame["observed_duration_s"].to_numpy(dtype=float)
    fill_event = frame["fill_event"].to_numpy(dtype=int)
    fill_time = frame["first_fill_time_s"].to_numpy(dtype=float)
    event = (
        fill_event.astype(bool)
        & (fill_time > start_seconds)
        & (fill_time <= end_seconds)
    )
    at_risk_at_start = observed_duration > start_seconds
    outcome_observed = event | (observed_duration >= end_seconds)
    mask = at_risk_at_start & outcome_observed
    return mask, event[mask].astype(int)


def cumulative_fill_probabilities(interval_hazards):
    """Convert conditional interval hazards into monotone cumulative fills."""
    hazards = np.asarray(interval_hazards, dtype=float)
    if hazards.ndim != 2 or hazards.shape[1] == 0:
        raise ValueError("interval_hazards must be a non-empty 2D matrix.")
    if not np.isfinite(hazards).all() or ((hazards < 0) | (hazards > 1)).any():
        raise ValueError("Interval hazards must be finite probabilities in [0, 1].")
    return 1 - np.cumprod(1 - hazards, axis=1)


def count_monotonicity_violations(cumulative_probability, tolerance=1e-12):
    """Count rows whose cumulative probability decreases with the horizon."""
    probability = np.asarray(cumulative_probability, dtype=float)
    if probability.ndim != 2:
        raise ValueError("cumulative_probability must be a 2D matrix.")
    return int((np.diff(probability, axis=1) < -tolerance).any(axis=1).sum())
