"""Cost-aware threshold selection utilities for fill decisions."""

from __future__ import annotations

import numpy as np


def classification_cost(y_true, predictions, *, false_positive_cost, false_negative_cost):
    """Return total and per-order cost under a binary error-cost matrix."""
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if len(y_true) == 0 or len(y_true) != len(predictions):
        raise ValueError("Targets and predictions must be non-empty and aligned.")
    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("Error costs must be non-negative.")
    false_positives = int(((predictions == 1) & (y_true == 0)).sum())
    false_negatives = int(((predictions == 0) & (y_true == 1)).sum())
    total = (
        false_positive_cost * false_positives
        + false_negative_cost * false_negatives
    )
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_cost": float(total),
        "cost_per_order": float(total / len(y_true)),
    }


def select_cost_threshold(
    y_true,
    probability,
    *,
    false_positive_cost,
    false_negative_cost,
    minimum_positive_rate=0.0,
    maximum_positive_rate=1.0,
):
    """Choose a validation threshold that minimizes the stated error cost.

    When several observed score boundaries have the same empirical cost, the
    boundary closest to the calibrated-probability Bayes threshold is used.
    """
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(probability):
        raise ValueError("Targets and probabilities must be non-empty and aligned.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if false_positive_cost <= 0 or false_negative_cost <= 0:
        raise ValueError("Threshold-selection costs must be positive.")
    if not 0 <= minimum_positive_rate <= maximum_positive_rate <= 1:
        raise ValueError("Positive-rate guardrails must satisfy 0 <= min <= max <= 1.")

    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, 1.0]),
                probability,
                np.array([np.nextafter(probability.max(), 1.0)]),
            ]
        )
    )
    bayes_threshold = false_positive_cost / (
        false_positive_cost + false_negative_cost
    )
    evaluated = []
    for threshold in candidates:
        predictions = (probability >= threshold).astype(int)
        positive_rate = float(predictions.mean())
        if not minimum_positive_rate <= positive_rate <= maximum_positive_rate:
            continue
        cost = classification_cost(
            y_true,
            predictions,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        evaluated.append(
            (
                cost["cost_per_order"],
                abs(float(threshold) - bayes_threshold),
                -float(threshold),
                float(threshold),
            )
        )
    if not evaluated:
        raise ValueError("No observed threshold satisfies the positive-rate guardrail.")
    _, _, _, selected_threshold = min(evaluated)
    return selected_threshold
