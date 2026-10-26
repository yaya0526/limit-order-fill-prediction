import numpy as np

from src.evaluation import (
    bootstrap_metric_intervals,
    expected_calibration_error,
    safe_probability_metrics,
)


def test_expected_calibration_error_is_zero_for_matching_bins():
    y_true = np.array([0, 0, 1, 1])
    probability = np.array([0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(y_true, probability, bins=2) == 0.0


def test_safe_probability_metrics_returns_consistent_confusion_counts():
    metrics = safe_probability_metrics(
        y_true=np.array([0, 0, 1, 1]),
        probability=np.array([0.10, 0.80, 0.60, 0.90]),
        threshold=0.5,
    )

    assert metrics["eligible_rows"] == 4
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 0, 2)
    assert metrics["predicted_fill_rate"] == 0.75
    assert metrics["recall"] == 1.0


def test_bootstrap_intervals_are_reproducible():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probability = np.array([0.05, 0.15, 0.35, 0.60, 0.80, 0.95])

    first = bootstrap_metric_intervals(
        y_true,
        probability,
        threshold=0.5,
        random_seed=7,
        resamples=30,
    )
    second = bootstrap_metric_intervals(
        y_true,
        probability,
        threshold=0.5,
        random_seed=7,
        resamples=30,
    )

    assert first == second
    assert first["pr_auc"]["ci_low"] <= first["pr_auc"]["ci_high"]
