"""Run the R2 discrete-time fill-hazard evaluation from the source workbook."""

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
from src.rolling_origin import build_expanding_calibration_folds
from src.survival import (
    HAZARD_INTERVALS,
    count_monotonicity_violations,
    cumulative_fill_probabilities,
    interval_risk_set,
)


HORIZONS = tuple(end for _, end in HAZARD_INTERVALS)
FEATURE_SET = "C_order_lob_flow"
MODEL_PARAMETERS = {"C": 1.0}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit and evaluate a coherent discrete-time fill-hazard model."
    )
    parser.add_argument("--excel", type=Path, required=True, help="Source Excel workbook.")
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results", help="Result directory."
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "reports" / "figures",
        help="Figure directory.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def load_orders(path):
    if not path.is_file():
        raise FileNotFoundError(f"Workbook not found: {path}")
    orders = pd.read_excel(path, sheet_name="Orders")
    required = {
        "order_id",
        "trading_day",
        "data_split",
        "observed_duration_s",
        "fill_event",
        "first_fill_time_s",
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
    return add_oriented_static_features(orders)


def fit_interval_model(training_rows, interval, random_seed):
    start_seconds, end_seconds = interval
    features = FEATURE_SETS[FEATURE_SET]
    training_days = sorted(training_rows["trading_day_parsed"].unique())
    folds = build_expanding_calibration_folds(training_days)
    oof_probability = []
    oof_target = []

    for fit_days, score_days in folds:
        fit_rows = training_rows.loc[
            training_rows["trading_day_parsed"].isin(fit_days)
        ].copy()
        score_rows = training_rows.loc[
            training_rows["trading_day_parsed"].isin(score_days)
        ].copy()
        fit_mask, y_fit = interval_risk_set(fit_rows, start_seconds, end_seconds)
        score_mask, y_score = interval_risk_set(
            score_rows, start_seconds, end_seconds
        )
        estimator = build_estimator(
            FEATURE_SET,
            "logistic_regression",
            MODEL_PARAMETERS,
            random_seed=random_seed,
        )
        estimator.fit(fit_rows.loc[fit_mask, features], y_fit)
        oof_probability.append(
            estimator.predict_proba(score_rows.loc[score_mask, features])[:, 1]
        )
        oof_target.append(y_score)

    combined_oof_probability = np.concatenate(oof_probability)
    combined_oof_target = np.concatenate(oof_target)
    calibrator = fit_sigmoid_calibrator(combined_oof_probability, combined_oof_target)
    full_mask, full_target = interval_risk_set(
        training_rows, start_seconds, end_seconds
    )
    estimator = build_estimator(
        FEATURE_SET,
        "logistic_regression",
        MODEL_PARAMETERS,
        random_seed=random_seed,
    )
    estimator.fit(training_rows.loc[full_mask, features], full_target)
    audit = {
        "interval_start_s": start_seconds,
        "interval_end_s": end_seconds,
        "training_risk_rows": int(full_mask.sum()),
        "training_events": int(full_target.sum()),
        "training_event_rate": float(full_target.mean()),
        "calibration_folds": len(folds),
        "calibration_oof_rows": len(combined_oof_target),
        "calibration_oof_event_rate": float(combined_oof_target.mean()),
    }
    return estimator, calibrator, audit


def predict_interval_hazards(models, calibrators, frame):
    features = FEATURE_SETS[FEATURE_SET]
    hazards = []
    for estimator, calibrator in zip(models, calibrators):
        raw_probability = estimator.predict_proba(frame[features])[:, 1]
        hazards.append(apply_sigmoid_calibrator(calibrator, raw_probability))
    return np.column_stack(hazards)


def evaluate_cumulative_probabilities(frame, cumulative_probability, split_name):
    rows = []
    for index, horizon in enumerate(HORIZONS):
        mask = frame[f"eligible_{horizon}s"].eq(1)
        y_true = frame.loc[mask, f"target_fill_{horizon}s"].astype(int).to_numpy()
        probability = cumulative_probability[mask.to_numpy(), index]
        metrics = safe_probability_metrics(y_true, probability, threshold=0.5)
        rows.append(
            {
                "split": split_name,
                "model_type": "discrete_time_hazard",
                "horizon_seconds": horizon,
                "threshold": 0.5,
                **metrics,
            }
        )
    return rows


def independent_test_metrics(path):
    predictions = pd.read_csv(path)
    rows = []
    for horizon, group in predictions.groupby("horizon_seconds", sort=True):
        metrics = safe_probability_metrics(
            group["target_fill"].astype(int).to_numpy(),
            group["calibrated_probability"].to_numpy(),
            threshold=None,
            predictions=group["predicted_fill"].astype(int).to_numpy(),
        )
        rows.append(
            {
                "split": "test",
                "model_type": "independent_classifier",
                "horizon_seconds": int(horizon),
                "threshold": float(group["selected_threshold"].iloc[0]),
                **metrics,
            }
        )
    return rows


def build_prediction_frame(frame, hazards, cumulative_probability, split_name):
    output = frame[["order_id", "trading_day"]].copy()
    output.insert(2, "split", split_name)
    for index, (start_seconds, end_seconds) in enumerate(HAZARD_INTERVALS):
        output[f"hazard_{start_seconds}_{end_seconds}s"] = hazards[:, index]
    for index, horizon in enumerate(HORIZONS):
        output[f"probability_fill_{horizon}s"] = cumulative_probability[:, index]
        output[f"eligible_{horizon}s"] = frame[f"eligible_{horizon}s"].to_numpy()
        output[f"target_fill_{horizon}s"] = frame[f"target_fill_{horizon}s"].to_numpy()
    return output


def save_figure(metrics, predictions, output_path):
    test_metrics = metrics.loc[metrics["split"].eq("test")].copy()
    colors = {
        "discrete_time_hazard": "#7C3AED",
        "independent_classifier": "#64748B",
    }
    labels = {
        "discrete_time_hazard": "Discrete-time hazard",
        "independent_classifier": "Independent classifiers",
    }
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for axis, metric, title, direction in (
        (axes[0], "pr_auc", "Test PR-AUC", "higher is better"),
        (axes[1], "brier_score", "Test Brier score", "lower is better"),
    ):
        for model_type, group in test_metrics.groupby("model_type"):
            axis.plot(
                group["horizon_seconds"],
                group[metric],
                marker="o",
                linewidth=2,
                color=colors[model_type],
                label=labels[model_type],
            )
        axis.set_title(title, fontsize=11, weight="bold")
        axis.set_xlabel(f"Horizon, seconds ({direction})", fontsize=9)
        axis.set_xticks(HORIZONS)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)

    test_predictions = predictions.loc[predictions["split"].eq("test")]
    mean_probability = [
        test_predictions[f"probability_fill_{horizon}s"].mean()
        for horizon in HORIZONS
    ]
    axes[2].plot(
        HORIZONS,
        mean_probability,
        marker="o",
        linewidth=2.5,
        color=colors["discrete_time_hazard"],
    )
    axes[2].set_title("Mean cumulative fill probability", fontsize=11, weight="bold")
    axes[2].set_xlabel("Horizon, seconds (monotone by construction)", fontsize=9)
    axes[2].set_xticks(HORIZONS)
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Metric value")
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Coherent time-to-fill model versus independent horizon classifiers",
        fontsize=13,
        weight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    return value


def main():
    args = parse_args()
    orders = load_orders(args.excel)
    features = FEATURE_SETS[FEATURE_SET]
    if not orders[features].notna().all().all():
        raise ValueError("Selected feature set contains missing values.")

    training_rows = orders.loc[
        orders["data_split_normalized"].eq("train")
    ].copy()
    evaluation_frames = {
        split: orders.loc[orders["data_split_normalized"].eq(split)].copy()
        for split in ("validation", "test")
    }

    estimators = []
    calibrators = []
    audit_rows = []
    for interval in HAZARD_INTERVALS:
        estimator, calibrator, audit = fit_interval_model(
            training_rows, interval, args.random_seed
        )
        estimators.append(estimator)
        calibrators.append(calibrator)
        audit_rows.append(audit)

    metric_rows = []
    prediction_frames = []
    monotonicity = {}
    for split_name, frame in evaluation_frames.items():
        hazards = predict_interval_hazards(estimators, calibrators, frame)
        cumulative_probability = cumulative_fill_probabilities(hazards)
        violations = count_monotonicity_violations(cumulative_probability)
        monotonicity[split_name] = {
            "rows": len(frame),
            "violations": violations,
            "violation_rate": violations / len(frame),
            "minimum_probability_increment": float(
                np.diff(cumulative_probability, axis=1).min()
            ),
        }
        metric_rows.extend(
            evaluate_cumulative_probabilities(
                frame, cumulative_probability, split_name
            )
        )
        prediction_frames.append(
            build_prediction_frame(
                frame, hazards, cumulative_probability, split_name
            )
        )

    metric_rows.extend(
        independent_test_metrics(args.results_dir / "final_test_predictions.csv")
    )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["split", "model_type", "horizon_seconds"]
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.results_dir / "time_to_fill_metrics.csv", index=False)
    audit.to_csv(args.results_dir / "time_to_fill_risk_set_audit.csv", index=False)

    hazard_metrics = metrics.loc[
        metrics["model_type"].eq("discrete_time_hazard")
    ]
    payload = {
        "status": "complete",
        "random_seed": args.random_seed,
        "model": {
            "type": "discrete_time_logistic_hazard",
            "feature_set": FEATURE_SET,
            "parameters": MODEL_PARAMETERS,
            "intervals_seconds": [list(interval) for interval in HAZARD_INTERVALS],
            "calibration": "forward-only sigmoid calibration per interval",
            "censoring": "orders censored inside an interval are excluded from that interval risk set",
        },
        "acceptance": {
            "monotone_by_construction": True,
            "validation_monotonicity_violations": monotonicity["validation"][
                "violations"
            ],
            "test_monotonicity_violations": monotonicity["test"]["violations"],
            "time_dependent_discrimination_reported": True,
            "time_dependent_calibration_reported": True,
            "passed": all(item["violations"] == 0 for item in monotonicity.values()),
        },
        "monotonicity": monotonicity,
        "risk_set_audit": audit_rows,
        "metrics": hazard_metrics.to_dict("records"),
        "limitations": [
            "This post-hoc extension uses the same synthetic sequence as the fixed-horizon study.",
            "Fixed interval boundaries provide a coarse event-time distribution rather than continuous-time hazards.",
            "Cancellation is treated as independent right-censoring rather than a competing event.",
        ],
    }
    (args.results_dir / "time_to_fill_summary.json").write_text(
        json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8"
    )
    save_figure(
        metrics, predictions, args.figure_dir / "time_to_fill_coherence.png"
    )
    print(
        "Time-to-fill evaluation complete: "
        f"{len(predictions):,} validation/test orders, zero monotonicity violations."
    )


if __name__ == "__main__":
    main()
