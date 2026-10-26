"""Run the R1 rolling-origin robustness evaluation from the source workbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import (
    apply_sigmoid_calibrator,
    build_estimator,
    fit_sigmoid_calibrator,
    safe_probability_metrics,
)
from src.modeling import FEATURE_SETS, add_oriented_static_features
from src.rolling_origin import (
    METRIC_NAMES,
    build_expanding_calibration_folds,
    build_rolling_origin_windows,
    day_block_bootstrap_metric_intervals,
    select_f1_threshold,
)


HORIZONS = (5, 30, 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate frozen fill-probability policies across rolling origins."
    )
    parser.add_argument("--excel", type=Path, required=True, help="Source Excel workbook.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "results" / "selected_model_policy.json",
        help="Frozen model policy JSON.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results", help="Result directory."
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "reports" / "figures",
        help="Figure directory.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def iso_date(value):
    return str(pd.Timestamp(value).date())


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return iso_date(value)
    return value


def load_inputs(excel_path, policy_path):
    if not excel_path.is_file():
        raise FileNotFoundError(f"Workbook not found: {excel_path}")
    orders = pd.read_excel(excel_path, sheet_name="Orders")
    required = {
        "order_id",
        "trading_day",
        "data_split",
        *(f"eligible_{horizon}s" for horizon in HORIZONS),
        *(f"target_fill_{horizon}s" for horizon in HORIZONS),
    }
    missing = required - set(orders.columns)
    if missing:
        raise ValueError(f"Orders sheet missing required columns: {sorted(missing)}")
    if len(orders) != 10_000 or not orders["order_id"].is_unique:
        raise ValueError("Expected 10,000 rows with unique order_id values.")

    orders["trading_day_parsed"] = pd.to_datetime(orders["trading_day"])
    orders["data_split_normalized"] = orders["data_split"].astype(str).str.lower()
    orders = add_oriented_static_features(orders)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    return orders, policy


def evaluate_window(orders, policy, window, horizon, random_seed):
    horizon_policy = policy["horizons"][str(horizon)]
    feature_set = horizon_policy["feature_set"]
    model_name = horizon_policy["model"]
    parameters = horizon_policy["parameters"]
    features = FEATURE_SETS[feature_set]
    eligibility = f"eligible_{horizon}s"
    target = f"target_fill_{horizon}s"

    train_rows = orders.loc[orders["trading_day_parsed"].isin(window["train_days"])].copy()
    validation_rows = orders.loc[
        orders["trading_day_parsed"].isin(window["validation_days"])
    ].copy()
    test_rows = orders.loc[orders["trading_day_parsed"].isin(window["test_days"])].copy()

    oof_probabilities = []
    oof_targets = []
    calibration_folds = build_expanding_calibration_folds(window["train_days"])
    for fit_days, score_days in calibration_folds:
        fit_mask = train_rows["trading_day_parsed"].isin(fit_days) & train_rows[
            eligibility
        ].eq(1)
        score_mask = train_rows["trading_day_parsed"].isin(score_days) & train_rows[
            eligibility
        ].eq(1)
        estimator = build_estimator(
            feature_set, model_name, parameters, random_seed=random_seed
        )
        estimator.fit(
            train_rows.loc[fit_mask, features],
            train_rows.loc[fit_mask, target].astype(int),
        )
        oof_probabilities.append(
            estimator.predict_proba(train_rows.loc[score_mask, features])[:, 1]
        )
        oof_targets.append(train_rows.loc[score_mask, target].astype(int).to_numpy())

    calibrator = fit_sigmoid_calibrator(
        np.concatenate(oof_probabilities), np.concatenate(oof_targets)
    )
    train_mask = train_rows[eligibility].eq(1)
    validation_mask = validation_rows[eligibility].eq(1)
    test_mask = test_rows[eligibility].eq(1)
    estimator = build_estimator(
        feature_set, model_name, parameters, random_seed=random_seed
    )
    estimator.fit(
        train_rows.loc[train_mask, features],
        train_rows.loc[train_mask, target].astype(int),
    )

    validation_probability = apply_sigmoid_calibrator(
        calibrator,
        estimator.predict_proba(validation_rows.loc[validation_mask, features])[:, 1],
    )
    y_validation = validation_rows.loc[validation_mask, target].astype(int).to_numpy()
    threshold = select_f1_threshold(y_validation, validation_probability)

    raw_test_probability = estimator.predict_proba(
        test_rows.loc[test_mask, features]
    )[:, 1]
    test_probability = apply_sigmoid_calibrator(calibrator, raw_test_probability)
    y_test = test_rows.loc[test_mask, target].astype(int).to_numpy()
    predictions = (test_probability >= threshold).astype(int)
    metrics = safe_probability_metrics(y_test, test_probability, threshold)

    metadata = {
        "window_id": window["window_id"],
        "horizon_seconds": horizon,
        "candidate": horizon_policy["candidate"],
        "feature_set": feature_set,
        "model": model_name,
        "calibration": "sigmoid",
        "threshold_rule": "maximize_validation_f1",
        "selected_threshold": threshold,
        "train_start": iso_date(min(window["train_days"])),
        "train_end": iso_date(max(window["train_days"])),
        "validation_start": iso_date(min(window["validation_days"])),
        "validation_end": iso_date(max(window["validation_days"])),
        "test_start": iso_date(min(window["test_days"])),
        "test_end": iso_date(max(window["test_days"])),
        "train_days": len(window["train_days"]),
        "calibration_folds": len(calibration_folds),
        "train_eligible_rows": int(train_mask.sum()),
        "validation_eligible_rows": int(validation_mask.sum()),
        **metrics,
    }

    prediction_frame = test_rows.loc[
        test_mask, ["order_id", "trading_day", "trading_day_parsed"]
    ].copy()
    prediction_frame.insert(0, "window_id", window["window_id"])
    prediction_frame["horizon_seconds"] = horizon
    prediction_frame["target_fill"] = y_test
    prediction_frame["raw_probability"] = raw_test_probability
    prediction_frame["calibrated_probability"] = test_probability
    prediction_frame["selected_threshold"] = threshold
    prediction_frame["predicted_fill"] = predictions
    return metadata, prediction_frame


def build_summary(window_metrics, predictions, resamples, random_seed):
    summary_rows = []
    summary_payload = {}
    for horizon in HORIZONS:
        window_slice = window_metrics.loc[
            window_metrics["horizon_seconds"].eq(horizon)
        ].copy()
        prediction_slice = predictions.loc[
            predictions["horizon_seconds"].eq(horizon)
        ].copy()
        pooled = safe_probability_metrics(
            prediction_slice["target_fill"].to_numpy(),
            prediction_slice["calibrated_probability"].to_numpy(),
            threshold=None,
            predictions=prediction_slice["predicted_fill"].to_numpy(),
        )
        intervals = day_block_bootstrap_metric_intervals(
            prediction_slice["target_fill"].to_numpy(),
            prediction_slice["calibrated_probability"].to_numpy(),
            prediction_slice["predicted_fill"].to_numpy(),
            prediction_slice["trading_day_parsed"].to_numpy(),
            random_seed=random_seed + horizon,
            resamples=resamples,
        )
        row = {
            "horizon_seconds": horizon,
            "window_count": len(window_slice),
            "unique_test_days": prediction_slice["trading_day_parsed"].nunique(),
            "pooled_eligible_rows": len(prediction_slice),
        }
        aggregates = {}
        for metric in METRIC_NAMES:
            values = window_slice[metric].astype(float)
            aggregate = {
                "window_mean": float(values.mean()),
                "window_std": float(values.std(ddof=1)),
                "window_min": float(values.min()),
                "window_max": float(values.max()),
                "pooled": float(pooled[metric]),
                "day_block_ci_low": intervals[metric]["ci_low"],
                "day_block_ci_high": intervals[metric]["ci_high"],
            }
            aggregates[metric] = aggregate
            for key, value in aggregate.items():
                row[f"{metric}_{key}"] = value
        summary_rows.append(row)
        summary_payload[str(horizon)] = {
            "candidate": window_slice.iloc[0]["candidate"],
            "feature_set": window_slice.iloc[0]["feature_set"],
            "model": window_slice.iloc[0]["model"],
            "aggregate_metrics": aggregates,
        }
    return pd.DataFrame(summary_rows), summary_payload


def save_figure(window_metrics, output_path):
    metric_specs = (
        ("pr_auc", "PR-AUC", "higher is better"),
        ("brier_score", "Brier score", "lower is better"),
        ("f1", "F1 at validation-selected threshold", "higher is better"),
    )
    colors = {5: "#2563EB", 30: "#D97706", 60: "#059669"}
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharex=True)
    for axis, (metric, title, direction) in zip(axes, metric_specs):
        for horizon in HORIZONS:
            subset = window_metrics.loc[
                window_metrics["horizon_seconds"].eq(horizon)
            ].sort_values("window_id")
            axis.plot(
                subset["window_id"],
                subset[metric],
                marker="o",
                linewidth=2,
                color=colors[horizon],
                label=f"{horizon}s",
            )
        axis.set_title(title, fontsize=11, weight="bold")
        axis.set_ylim(0, 1)
        axis.set_xlabel(f"Rolling test window ({direction})", fontsize=9)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(frameon=False, loc="best")
    figure.suptitle(
        "Rolling-origin robustness across three non-overlapping test windows",
        fontsize=13,
        weight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    orders, policy = load_inputs(args.excel, args.policy)
    selected_features = {
        policy["horizons"][str(horizon)]["feature_set"] for horizon in HORIZONS
    }
    missing_features = {
        feature
        for feature_set in selected_features
        for feature in FEATURE_SETS[feature_set]
        if feature not in orders.columns
    }
    if missing_features:
        raise ValueError(f"Workbook missing policy features: {sorted(missing_features)}")

    development_days = sorted(
        orders.loc[
            ~orders["data_split_normalized"].eq("test"), "trading_day_parsed"
        ].unique()
    )
    final_test_days = sorted(
        orders.loc[
            orders["data_split_normalized"].eq("test"), "trading_day_parsed"
        ].unique()
    )
    windows = build_rolling_origin_windows(development_days)
    if any(set(window["test_days"]) & set(final_test_days) for window in windows):
        raise ValueError("Rolling windows must not touch the original final test period.")

    metric_rows = []
    prediction_frames = []
    for window in windows:
        for horizon in HORIZONS:
            metrics, predictions = evaluate_window(
                orders, policy, window, horizon, args.random_seed
            )
            metric_rows.append(metrics)
            prediction_frames.append(predictions)

    window_metrics = pd.DataFrame(metric_rows).sort_values(
        ["horizon_seconds", "window_id"]
    )
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["horizon_seconds", "window_id", "trading_day_parsed", "order_id"]
    )
    summary, summary_payload = build_summary(
        window_metrics, predictions, args.bootstrap_resamples, args.random_seed
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    window_metrics.to_csv(
        args.results_dir / "rolling_origin_window_metrics.csv", index=False
    )
    summary.to_csv(args.results_dir / "rolling_origin_summary.csv", index=False)
    predictions.drop(columns="trading_day_parsed").to_csv(
        args.results_dir / "rolling_origin_predictions.csv", index=False
    )

    window_payload = []
    for window in windows:
        window_payload.append(
            {
                "window_id": window["window_id"],
                "train_start": iso_date(min(window["train_days"])),
                "train_end": iso_date(max(window["train_days"])),
                "validation_start": iso_date(min(window["validation_days"])),
                "validation_end": iso_date(max(window["validation_days"])),
                "test_start": iso_date(min(window["test_days"])),
                "test_end": iso_date(max(window["test_days"])),
            }
        )
    payload = {
        "status": "complete",
        "random_seed": args.random_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "design": {
            "policy": "Frozen feature set, model family, and hyperparameters from v0.1.0.",
            "calibration": "Forward-only sigmoid calibration fitted from training-period out-of-fold predictions.",
            "threshold": "F1-maximizing threshold selected on each window's validation period.",
            "uncertainty": "Trading-day block bootstrap over the six pooled rolling test days.",
            "original_final_test_untouched": True,
            "original_final_test_days": [iso_date(day) for day in final_test_days],
            "limitation": "This is a post-selection robustness audit: v0.1.0 hyperparameters were fixed before the rolling analysis, but were originally chosen using the broader development period.",
        },
        "acceptance": {
            "required_non_overlapping_test_windows": 3,
            "observed_non_overlapping_test_windows": len(windows),
            "dispersion_reported_with_mean": True,
            "passed": len(windows) >= 3,
        },
        "windows": window_payload,
        "horizons": summary_payload,
    }
    (args.results_dir / "rolling_origin_summary.json").write_text(
        json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8"
    )
    save_figure(
        window_metrics, args.figure_dir / "rolling_origin_robustness.png"
    )
    print(
        "Rolling-origin evaluation complete: "
        f"{len(windows)} windows, {len(predictions):,} eligible predictions."
    )


if __name__ == "__main__":
    main()
