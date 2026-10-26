import numpy as np

from src.rolling_origin import (
    build_expanding_calibration_folds,
    build_rolling_origin_windows,
    day_block_bootstrap_metric_intervals,
    select_f1_threshold,
)


def test_rolling_windows_are_forward_only_and_tests_do_not_overlap():
    windows = build_rolling_origin_windows(range(17))

    assert [len(window["train_days"]) for window in windows] == [8, 10, 12]
    assert [window["test_days"] for window in windows] == [
        (10, 11),
        (12, 13),
        (14, 15),
    ]
    all_test_days = [day for window in windows for day in window["test_days"]]
    assert len(all_test_days) == len(set(all_test_days))
    for window in windows:
        assert max(window["train_days"]) < min(window["validation_days"])
        assert max(window["validation_days"]) < min(window["test_days"])


def test_calibration_folds_only_score_future_days():
    folds = build_expanding_calibration_folds(range(12))

    assert len(folds) == 3
    for fit_days, score_days in folds:
        assert max(fit_days) < min(score_days)
        assert len(score_days) == 2


def test_f1_threshold_selects_a_valid_score_boundary():
    probability = np.array([0.10, 0.20, 0.70, 0.90])
    threshold = select_f1_threshold(np.array([0, 0, 1, 1]), probability)

    assert threshold in probability
    assert threshold == 0.70


def test_day_block_bootstrap_is_reproducible():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    probability = np.array([0.1, 0.8, 0.3, 0.7, 0.2, 0.9])
    predictions = (probability >= 0.5).astype(int)
    days = np.array([1, 1, 2, 2, 3, 3])

    first = day_block_bootstrap_metric_intervals(
        y_true,
        probability,
        predictions,
        days,
        random_seed=9,
        resamples=30,
    )
    second = day_block_bootstrap_metric_intervals(
        y_true,
        probability,
        predictions,
        days,
        random_seed=9,
        resamples=30,
    )

    assert first == second
    assert first["pr_auc"]["ci_low"] <= first["pr_auc"]["ci_high"]
