from datetime import datetime, timedelta

import pytest

from src.history_features import linear_trend, summarize_history


def make_order():
    submission = datetime(2022, 9, 1, 10, 0, 0)
    event_types = ("limit_submission", "cancellation", "market_trade", "modification")
    history = []
    for index in range(50):
        milliseconds_before_submission = index - 50
        mid_price = 100.0 + index * 0.001
        history.append({
            "sequence_step": index + 1,
            "milliseconds_before_submission": milliseconds_before_submission,
            "event_timestamp": (
                submission + timedelta(milliseconds=milliseconds_before_submission)
            ).isoformat(),
            "mid_price": mid_price,
            "spread_ticks": 1 + index % 2,
            "five_level_bid_ask_imbalance": -0.2 + index * 0.01,
            "microprice": mid_price + 0.0005,
            "bid_prices": [99.99 - 0.01 * level for level in range(5)],
            "ask_prices": [100.01 + 0.01 * level for level in range(5)],
            "bid_volumes": [100 + index + level for level in range(5)],
            "ask_volumes": [120 + index + level for level in range(5)],
            "event_type": event_types[index % len(event_types)],
            "event_side": "bid" if index % 2 == 0 else "ask",
        })
    return {
        "order_id": "order-1",
        "submission_timestamp": submission.isoformat(),
        "history": history,
    }


def test_linear_trend_recovers_known_slope():
    assert linear_trend([2, 4, 6, 8]) == pytest.approx(2.0)


def test_summarize_history_preserves_chronology_checks():
    summary = summarize_history(make_order())

    assert summary["order_id"] == "order-1"
    assert summary["history_snapshot_count"] == 50
    assert summary["check_sequence_1_to_50"] is True
    assert summary["check_offsets_strictly_negative"] is True
    assert summary["check_offsets_strictly_increasing"] is True
    assert summary["check_event_timestamps_before_submission"] is True
    assert summary["check_event_timestamps_increasing"] is True
    assert summary["lookback_span_ms"] == 49.0
    assert summary["last_event_to_submission_ms"] == 1.0
