"""Utilities for leakage-aware rolling-origin robustness evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

try:  # Package import used by tests and external callers.
    from .evaluation import safe_probability_metrics
except ImportError:  # Script compatibility when src/ is added to sys.path.
    from evaluation import safe_probability_metrics


METRIC_NAMES = (
    "true_fill_rate",
    "predicted_fill_rate",
    "mean_predicted_probability",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "ece_10",
    "accuracy",
    "precision",
    "recall",
    "f1",
)


def build_rolling_origin_windows(
    days,
    *,
    initial_train_days=8,
    validation_days=2,
    test_days=2,
    window_count=3,
    step_days=2,
):
    """Build expanding chronological windows with disjoint test periods."""
    ordered_days = tuple(sorted(set(days)))
    required_days = (
        initial_train_days
        + (window_count - 1) * step_days
        + validation_days
        + test_days
    )
    if len(ordered_days) < required_days:
        raise ValueError(
            f"Need at least {required_days} distinct days, found {len(ordered_days)}."
        )
    if min(initial_train_days, validation_days, test_days, window_count, step_days) <= 0:
        raise ValueError("Window sizes and counts must be positive.")
    if step_days < test_days:
        raise ValueError("step_days must be at least test_days to keep tests disjoint.")

    windows = []
    all_test_days = []
    for index in range(window_count):
        train_end = initial_train_days + index * step_days
        validation_end = train_end + validation_days
        test_end = validation_end + test_days
        window = {
            "window_id": f"W{index + 1}",
            "train_days": ordered_days[:train_end],
            "validation_days": ordered_days[train_end:validation_end],
            "test_days": ordered_days[validation_end:test_end],
        }
        if not (
            max(window["train_days"]) < min(window["validation_days"])
            and max(window["validation_days"]) < min(window["test_days"])
        ):
            raise ValueError("Each rolling window must preserve chronological order.")
        windows.append(window)
        all_test_days.extend(window["test_days"])

    if len(all_test_days) != len(set(all_test_days)):
        raise ValueError("Rolling-origin test windows must not overlap.")
    return windows


def build_expanding_calibration_folds(days, *, min_fit_days=6, score_days=2):
    """Create forward-only folds used to fit a probability calibrator."""
    ordered_days = tuple(sorted(set(days)))
    if min_fit_days <= 0 or score_days <= 0:
        raise ValueError("Calibration fold sizes must be positive.")
    if len(ordered_days) < min_fit_days + score_days:
        raise ValueError("Not enough training days for an out-of-fold calibration block.")

    folds = []
    for score_start in range(min_fit_days, len(ordered_days), score_days):
        score_end = score_start + score_days
        if score_end > len(ordered_days):
            break
        folds.append((ordered_days[:score_start], ordered_days[score_start:score_end]))
    return folds


def select_f1_threshold(y_true, probability):
    """Select the threshold that maximizes validation F1."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(probability):
        raise ValueError("Targets and probabilities must be non-empty and aligned.")

    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        raise ValueError("Threshold selection requires at least one score boundary.")
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best_index = int(np.nanargmax(f1))
    return float(thresholds[best_index])


def day_block_bootstrap_metric_intervals(
    y_true,
    probability,
    predictions,
    trading_days,
    *,
    random_seed,
    resamples=1000,
    confidence=0.95,
):
    """Bootstrap whole trading days to retain within-day dependence."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    trading_days = np.asarray(trading_days)
    lengths = {len(y_true), len(probability), len(predictions), len(trading_days)}
    if len(lengths) != 1 or not y_true.size:
        raise ValueError("Bootstrap inputs must be non-empty and aligned.")
    if resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("resamples must be positive and confidence must be in (0, 1).")

    unique_days = np.unique(trading_days)
    day_indices = {day: np.flatnonzero(trading_days == day) for day in unique_days}
    rng = np.random.default_rng(random_seed)
    samples = {metric: [] for metric in METRIC_NAMES}

    for _ in range(resamples):
        sampled_days = rng.choice(unique_days, size=len(unique_days), replace=True)
        sampled_indices = np.concatenate([day_indices[day] for day in sampled_days])
        metrics = safe_probability_metrics(
            y_true[sampled_indices],
            probability[sampled_indices],
            threshold=None,
            predictions=predictions[sampled_indices],
        )
        for metric in METRIC_NAMES:
            if np.isfinite(metrics[metric]):
                samples[metric].append(metrics[metric])

    tail = (1 - confidence) / 2
    intervals = {}
    for metric, values in samples.items():
        values = np.asarray(values, dtype=float)
        intervals[metric] = {
            "ci_low": float(np.quantile(values, tail)),
            "ci_high": float(np.quantile(values, 1 - tail)),
        }
    return intervals
