import numpy as np
import pandas as pd

from src.market_simulation import (
    SCENARIOS,
    scenario_manifest_hash,
    simulate_scenario,
    transfer_status,
)
from src.modeling import FEATURE_SETS, add_oriented_static_features


def test_simulator_is_deterministic_for_same_seed():
    first = simulate_scenario(SCENARIOS[0], rows=150, random_seed=17)
    second = simulate_scenario(SCENARIOS[0], rows=150, random_seed=17)
    pd.testing.assert_frame_equal(first, second)


def test_simulator_provides_all_frozen_policy_features():
    frame = add_oriented_static_features(
        simulate_scenario(SCENARIOS[0], rows=150, random_seed=23)
    )
    required = set(FEATURE_SETS["B_order_lob"]) | set(
        FEATURE_SETS["C_order_lob_flow"]
    )
    assert required <= set(frame.columns)
    assert not frame[list(required)].isna().any().any()


def test_cumulative_fill_labels_are_monotone():
    for scenario in SCENARIOS:
        frame = simulate_scenario(scenario, rows=150, random_seed=31)
        assert np.all(frame["target_fill_5s"] <= frame["target_fill_30s"])
        assert np.all(frame["target_fill_30s"] <= frame["target_fill_60s"])


def test_scenario_declaration_and_transfer_rule_are_auditable():
    assert len(SCENARIOS) == 4
    assert len({scenario.name for scenario in SCENARIOS}) == len(SCENARIOS)
    assert len(scenario_manifest_hash()) == 64
    assert transfer_status(0.03, 0.10) == "ranking_and_calibration_transfer"
    assert transfer_status(0.03, -0.10) == "ranking_only_transfer"
    assert transfer_status(0.01, 0.20) == "failure"
    assert transfer_status(float("nan"), float("nan")) == "failure"
