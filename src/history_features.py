"""Reusable construction of pre-submission LOB history features."""

from collections import Counter
from pathlib import Path
import gzip

import ijson
import numpy as np
import pandas as pd


TICK_SIZE = 0.01
EVENT_TYPES = (
    "limit_submission",
    "cancellation",
    "market_trade",
    "modification",
)
EVENT_SIDES = ("bid", "ask")


FEATURE_GROUPS = {
    "price_movement": [
        "mid_return_1_50_bps",
        "best_bid_change_count",
        "best_ask_change_count",
        "oriented_net_best_quote_direction_ticks",
    ],
    "spread_behavior": [
        "spread_mean_50",
        "spread_max_50",
        "spread_final",
        "spread_one_tick_fraction",
    ],
    "imbalance_behavior": [
        "oriented_imbalance_mean_50",
        "imbalance_std_50",
        "oriented_imbalance_min_50",
        "oriented_imbalance_max_50",
        "oriented_imbalance_final",
        "oriented_imbalance_trend_50",
    ],
    "depth_change": [
        "same_side_depth_change_1_50",
        "opposite_side_depth_change_1_50",
    ],
    "microprice_signal": [
        "micro_gap_final_bps",
        "micro_gap_mean_50_bps",
        "micro_gap_trend_50",
    ],
    "event_composition": [
        f"event_count_{event_type}_{orientation}"
        for event_type in EVENT_TYPES
        for orientation in ("same_side", "opposite_side")
    ],
    "event_timing": [
        "lookback_span_ms",
        "mean_inter_event_ms",
        "last_event_to_submission_ms",
    ],
}

CONSTRUCTED_HISTORY_FEATURES = [
    feature
    for features in FEATURE_GROUPS.values()
    for feature in features
]

# The final snapshot is exactly 1 ms before submission for every supplied order.
# Keep this variable for chronology audits, but exclude it from prediction.
AUDIT_ONLY_FEATURES = ["last_event_to_submission_ms"]
MODEL_HISTORY_FEATURES = [
    feature
    for feature in CONSTRUCTED_HISTORY_FEATURES
    if feature not in AUDIT_ONLY_FEATURES
]


def open_json_binary(path):
    path = Path(path)
    if path.name.endswith(".json.gz"):
        return gzip.open(path, mode="rb")
    return path.open(mode="rb")


def linear_trend(values):
    """OLS slope against sequence steps 1..n using a centered formula."""
    y = np.asarray(values, dtype=float)
    x = np.arange(1, len(y) + 1, dtype=float)
    x_centered = x - x.mean()
    return float(np.dot(x_centered, y - y.mean()) / np.dot(x_centered, x_centered))


def summarize_history(order, tick_size=TICK_SIZE):
    history = order.get("history", [])
    sequence_steps = [snapshot.get("sequence_step") for snapshot in history]
    offsets_ms = np.asarray(
        [float(snapshot["milliseconds_before_submission"]) for snapshot in history],
        dtype=float,
    )

    submission_time = pd.Timestamp(order["submission_timestamp"])
    event_times = pd.DatetimeIndex(
        [pd.Timestamp(snapshot["event_timestamp"]) for snapshot in history]
    )

    mid = np.asarray([float(snapshot["mid_price"]) for snapshot in history])
    spread = np.asarray([float(snapshot["spread_ticks"]) for snapshot in history])
    imbalance = np.asarray(
        [float(snapshot["five_level_bid_ask_imbalance"]) for snapshot in history]
    )
    microprice = np.asarray(
        [float(snapshot["microprice"]) for snapshot in history]
    )
    micro_gap_bps = 10_000 * (microprice - mid) / mid

    bid_l1 = np.asarray([float(snapshot["bid_prices"][0]) for snapshot in history])
    ask_l1 = np.asarray([float(snapshot["ask_prices"][0]) for snapshot in history])
    bid_depth = np.asarray(
        [sum(float(value) for value in snapshot["bid_volumes"]) for snapshot in history]
    )
    ask_depth = np.asarray(
        [sum(float(value) for value in snapshot["ask_volumes"]) for snapshot in history]
    )

    event_type_counts = Counter(snapshot["event_type"] for snapshot in history)
    event_type_side_counts = Counter(
        (snapshot["event_type"], snapshot["event_side"])
        for snapshot in history
    )

    row = {
        "order_id": str(order["order_id"]),
        "history_snapshot_count": len(history),
        "check_sequence_1_to_50": sequence_steps == list(range(1, 51)),
        "check_offsets_strictly_negative": bool(np.all(offsets_ms < 0)),
        "check_offsets_strictly_increasing": bool(np.all(np.diff(offsets_ms) > 0)),
        "check_event_timestamps_before_submission": bool(np.all(event_times < submission_time)),
        "check_event_timestamps_increasing": bool(np.all(event_times[1:] > event_times[:-1])),
        "mid_return_1_50_bps": 10_000 * (mid[-1] - mid[0]) / mid[0],
        "best_bid_change_count": int(np.count_nonzero(np.diff(bid_l1))),
        "best_ask_change_count": int(np.count_nonzero(np.diff(ask_l1))),
        "net_best_quote_direction_ticks": float(
            np.round(((bid_l1[-1] - bid_l1[0]) + (ask_l1[-1] - ask_l1[0])) / tick_size)
        ),
        "spread_mean_50": float(spread.mean()),
        "spread_max_50": float(spread.max()),
        "spread_final": float(spread[-1]),
        "spread_one_tick_fraction": float(np.mean(spread == 1)),
        "imbalance_mean_50": float(imbalance.mean()),
        "imbalance_std_50": float(imbalance.std(ddof=0)),
        "imbalance_min_50": float(imbalance.min()),
        "imbalance_max_50": float(imbalance.max()),
        "imbalance_final": float(imbalance[-1]),
        "imbalance_trend_50": linear_trend(imbalance),
        "bid_depth_change_1_50": float(bid_depth[-1] - bid_depth[0]),
        "ask_depth_change_1_50": float(ask_depth[-1] - ask_depth[0]),
        "micro_gap_final_bps": float(micro_gap_bps[-1]),
        "micro_gap_mean_50_bps": float(micro_gap_bps.mean()),
        "micro_gap_trend_50": linear_trend(micro_gap_bps),
        "lookback_span_ms": float(offsets_ms[-1] - offsets_ms[0]),
        "mean_inter_event_ms": float(np.diff(offsets_ms).mean()),
        "last_event_to_submission_ms": float(-offsets_ms[-1]),
    }

    for event_type in EVENT_TYPES:
        row[f"event_count_{event_type}"] = int(event_type_counts[event_type])
        for event_side in EVENT_SIDES:
            row[f"event_count_{event_type}_{event_side}"] = int(
                event_type_side_counts[(event_type, event_side)]
            )

    return row


def build_history_features(json_path):
    """Stream one JSON order at a time and return one feature row per order."""
    rows = []
    with open_json_binary(json_path) as file:
        for order in ijson.items(file, "orders.item"):
            rows.append(summarize_history(order))
    return pd.DataFrame(rows)


def add_order_side_features(frame):
    """Add buy/sell-oriented historical variables without dropping audit columns."""
    result = frame.copy()
    buy_mask = result["side"].astype(str).str.lower().eq("buy")
    side_sign = np.where(buy_mask, 1.0, -1.0)

    result["oriented_net_best_quote_direction_ticks"] = (
        result["net_best_quote_direction_ticks"] * side_sign
    )
    result["oriented_imbalance_mean_50"] = result["imbalance_mean_50"] * side_sign
    result["oriented_imbalance_final"] = result["imbalance_final"] * side_sign
    result["oriented_imbalance_trend_50"] = result["imbalance_trend_50"] * side_sign
    result["oriented_imbalance_min_50"] = np.where(
        buy_mask,
        result["imbalance_min_50"],
        -result["imbalance_max_50"],
    )
    result["oriented_imbalance_max_50"] = np.where(
        buy_mask,
        result["imbalance_max_50"],
        -result["imbalance_min_50"],
    )

    result["same_side_depth_change_1_50"] = np.where(
        buy_mask,
        result["bid_depth_change_1_50"],
        result["ask_depth_change_1_50"],
    )
    result["opposite_side_depth_change_1_50"] = np.where(
        buy_mask,
        result["ask_depth_change_1_50"],
        result["bid_depth_change_1_50"],
    )

    for event_type in EVENT_TYPES:
        result[f"event_count_{event_type}_same_side"] = np.where(
            buy_mask,
            result[f"event_count_{event_type}_bid"],
            result[f"event_count_{event_type}_ask"],
        )
        result[f"event_count_{event_type}_opposite_side"] = np.where(
            buy_mask,
            result[f"event_count_{event_type}_ask"],
            result[f"event_count_{event_type}_bid"],
        )

    return result
