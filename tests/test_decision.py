import numpy as np

from src.decision import classification_cost, select_cost_threshold


def test_classification_cost_uses_stated_error_weights():
    cost = classification_cost(
        y_true=np.array([0, 0, 1, 1]),
        predictions=np.array([0, 1, 0, 1]),
        false_positive_cost=3,
        false_negative_cost=1,
    )

    assert cost["false_positives"] == 1
    assert cost["false_negatives"] == 1
    assert cost["total_cost"] == 4
    assert cost["cost_per_order"] == 1


def test_precision_first_cost_selects_higher_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probability = np.array([0.20, 0.35, 0.55, 0.45, 0.65, 0.80])

    balanced = select_cost_threshold(
        y_true,
        probability,
        false_positive_cost=1,
        false_negative_cost=1,
    )
    precision_first = select_cost_threshold(
        y_true,
        probability,
        false_positive_cost=3,
        false_negative_cost=1,
    )

    assert precision_first >= balanced


def test_threshold_selection_is_reproducible():
    y_true = np.array([0, 1, 0, 1])
    probability = np.array([0.1, 0.6, 0.4, 0.9])

    first = select_cost_threshold(
        y_true,
        probability,
        false_positive_cost=2,
        false_negative_cost=1,
    )
    second = select_cost_threshold(
        y_true,
        probability,
        false_positive_cost=2,
        false_negative_cost=1,
    )

    assert first == second


def test_positive_rate_guardrail_prevents_no_action_solution():
    y_true = np.array([0, 0, 0, 1])
    probability = np.array([0.1, 0.2, 0.3, 0.4])

    threshold = select_cost_threshold(
        y_true,
        probability,
        false_positive_cost=10,
        false_negative_cost=1,
        minimum_positive_rate=0.20,
        maximum_positive_rate=0.80,
    )
    positive_rate = (probability >= threshold).mean()

    assert 0.20 <= positive_rate <= 0.80
