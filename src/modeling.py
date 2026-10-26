"""Shared feature-set and preprocessing utilities for fill prediction."""

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # Package import used by tests and external callers.
    from .history_features import MODEL_HISTORY_FEATURES
except ImportError:  # Notebook compatibility when src/ is added to sys.path.
    from history_features import MODEL_HISTORY_FEATURES


A_NUMERIC = [
    "seconds_from_open",
    "order_age_s",
    "order_size",
    "price_distance_ticks",
    "quantity_filled_before_t0",
    "queue_ahead_qty",
    "same_price_total_qty",
    "normalized_queue_ahead",
]
A_CATEGORICAL = ["side", "session_phase"]

LOB_LEVEL_COLUMNS = []
for level in range(1, 6):
    LOB_LEVEL_COLUMNS.extend([
        f"bid_price_l{level}",
        f"bid_volume_l{level}",
        f"ask_price_l{level}",
        f"ask_volume_l{level}",
    ])

B_ADDITIONAL_NUMERIC = LOB_LEVEL_COLUMNS + [
    "mid_price",
    "spread_ticks",
    "relative_spread_bps",
    "oriented_top_level_imbalance",
    "five_level_side_imbalance",
    "same_side_depth_5",
    "opposite_side_depth_5",
    "microprice",
    "volatility_60s_bps",
]

C_ADDITIONAL_NUMERIC = [
    "same_side_submissions_10s",
    "opposite_side_submissions_10s",
    "same_side_cancellations_10s",
    "opposite_side_cancellations_10s",
    "modifications_10s",
    "aggressive_trade_volume_against_order_10s",
    "aggressive_trade_volume_with_order_side_10s",
    "trade_count_10s",
    "time_since_last_trade_ms",
]
C_ADDITIONAL_CATEGORICAL = ["market_activity_group"]

FEATURE_SETS = {
    "A_order": A_NUMERIC + A_CATEGORICAL,
    "B_order_lob": A_NUMERIC + A_CATEGORICAL + B_ADDITIONAL_NUMERIC,
    "C_order_lob_flow": (
        A_NUMERIC
        + A_CATEGORICAL
        + B_ADDITIONAL_NUMERIC
        + C_ADDITIONAL_NUMERIC
        + C_ADDITIONAL_CATEGORICAL
    ),
    "D_full_history": (
        A_NUMERIC
        + A_CATEGORICAL
        + B_ADDITIONAL_NUMERIC
        + C_ADDITIONAL_NUMERIC
        + C_ADDITIONAL_CATEGORICAL
        + MODEL_HISTORY_FEATURES
    ),
}

CATEGORICAL_FEATURES = set(A_CATEGORICAL + C_ADDITIONAL_CATEGORICAL)


def add_oriented_static_features(frame):
    result = frame.copy()
    buy_mask = result["side"].astype(str).str.lower().eq("buy")
    result["oriented_top_level_imbalance"] = np.where(
        buy_mask,
        result["top_level_imbalance"],
        -result["top_level_imbalance"],
    )
    return result


def build_preprocessor(features):
    categorical = [feature for feature in features if feature in CATEGORICAL_FEATURES]
    numeric = [feature for feature in features if feature not in CATEGORICAL_FEATURES]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ], remainder="drop", sparse_threshold=0)


def probability_metrics(y_true, probabilities, threshold=0.5):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }
