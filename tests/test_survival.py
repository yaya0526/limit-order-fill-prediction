import numpy as np
import pandas as pd

from src.survival import (
    count_monotonicity_violations,
    cumulative_fill_probabilities,
    interval_risk_set,
)


def test_interval_risk_set_excludes_mid_interval_censoring():
    frame = pd.DataFrame(
        {
            "observed_duration_s": [3.0, 10.0, 20.0, 40.0],
            "fill_event": [0, 1, 0, 1],
            "first_fill_time_s": [np.nan, 10.0, np.nan, 40.0],
        }
    )

    mask, target = interval_risk_set(frame, 5, 30)

    assert mask.tolist() == [False, True, False, True]
    assert target.tolist() == [1, 0]


def test_interval_boundary_fill_belongs_to_earlier_interval():
    frame = pd.DataFrame(
        {
            "observed_duration_s": [5.0],
            "fill_event": [1],
            "first_fill_time_s": [5.0],
        }
    )

    first_mask, first_target = interval_risk_set(frame, 0, 5)
    second_mask, second_target = interval_risk_set(frame, 5, 30)

    assert first_mask.tolist() == [True]
    assert first_target.tolist() == [1]
    assert second_mask.tolist() == [False]
    assert second_target.tolist() == []


def test_cumulative_fill_probabilities_are_monotone_and_correct():
    hazards = np.array([[0.10, 0.20, 0.30], [0.50, 0.10, 0.10]])
    cumulative = cumulative_fill_probabilities(hazards)

    np.testing.assert_allclose(
        cumulative,
        np.array([[0.10, 0.28, 0.496], [0.50, 0.55, 0.595]]),
    )
    assert count_monotonicity_violations(cumulative) == 0


def test_monotonicity_audit_detects_decrease():
    probability = np.array([[0.1, 0.2, 0.3], [0.4, 0.35, 0.5]])

    assert count_monotonicity_violations(probability) == 1
