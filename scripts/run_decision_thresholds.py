"""Run the R3 validation-selected, cost-aware threshold analysis."""

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

from src.decision import classification_cost, select_cost_threshold
from src.evaluation import (
    apply_sigmoid_calibrator,
    build_estimator,
    fit_sigmoid_calibrator,
    safe_probability_metrics,
)
from src.modeling import FEATURE_SETS, add_oriented_static_features
from src.rolling_origin import build_expanding_calibration_folds


HORIZONS = (5, 30, 60)
SCENARIOS = {
    "precision_first": {
        "false_positive_cost": 3.0,
        "false_negative_cost": 1.0,
        "interpretation": "Waiting/opportunity cost is three times missed-fill cost.",
    },
    "balanced": {
        "false_positive_cost": 1.0,
        "false_negative_cost": 1.0,
        "interpretation": "False positives and false negatives have equal cost.",
    },
    "fill_seeking": {
        "false_positive_cost": 1.0,
        "false_negative_cost": 3.0,
        "interpretation": "Missed-fill cost is three times waiting/opportunity cost.",
    },
}
PRIMARY_SCENARIO = "precision_first"
VALIDATION_POSITIVE_FLOOR = 0.01
SELECTIVITY_CEILING = 0.80


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select and evaluate validation-only cost-aware thresholds."
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
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def load_inputs(excel_path, policy_path):
    if not excel_path.is_file():
        raise FileNotFoundError(f"Workbook not found: {excel_path}")
    orders = pd.read_excel(excel_path, sheet_name="Orders")
    orders["trading_day_parsed"] = pd.to_datetime(orders["trading_day"])
    orders["data_split_normalized"] = orders["data_split"].astype(str).str.lower()
    if len(orders) != 10_000 or not orders["order_id"].is_unique:
        raise ValueError("Expected 10,000 rows with unique order_id values.")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    return add_oriented_static_features(orders), policy


def reconstruct_probabilities(orders, horizon_policy, horizon, random_seed):
    feature_set = horizon_policy["feature_set"]
    model_name = horizon_policy["model"]
    parameters = horizon_policy["parameters"]
    features = FEATURE_SETS[feature_set]
    eligibility = f"eligible_{horizon}s"
    target = f"target_fill_{horizon}s"
    training_rows = orders.loc[
        orders["data_split_normalized"].eq("train")
    ].copy()
    training_days = sorted(training_rows["trading_day_parsed"].unique())
    folds = build_expanding_calibration_folds(training_days)

    oof_probability = []
    oof_target = []
    for fit_days, score_days in folds:
        fit_mask = training_rows["trading_day_parsed"].isin(fit_days) & training_rows[
            eligibility
        ].eq(1)
        score_mask = training_rows["trading_day_parsed"].isin(
            score_days
        ) & training_rows[eligibility].eq(1)
        estimator = build_estimator(
            feature_set, model_name, parameters, random_seed=random_seed
        )
        estimator.fit(
            training_rows.loc[fit_mask, features],
            training_rows.loc[fit_mask, target].astype(int),
        )
        oof_probability.append(
            estimator.predict_proba(training_rows.loc[score_mask, features])[:, 1]
        )
        oof_target.append(training_rows.loc[score_mask, target].astype(int).to_numpy())

    calibrator = fit_sigmoid_calibrator(
        np.concatenate(oof_probability), np.concatenate(oof_target)
    )
    train_mask = training_rows[eligibility].eq(1)
    estimator = build_estimator(
        feature_set, model_name, parameters, random_seed=random_seed
    )
    estimator.fit(
        training_rows.loc[train_mask, features],
        training_rows.loc[train_mask, target].astype(int),
    )

    output = {}
    for split in ("validation", "test"):
        rows = orders.loc[orders["data_split_normalized"].eq(split)].copy()
        mask = rows[eligibility].eq(1)
        probability = apply_sigmoid_calibrator(
            calibrator, estimator.predict_proba(rows.loc[mask, features])[:, 1]
        )
        output[split] = {
            "order_id": rows.loc[mask, "order_id"].to_numpy(),
            "y_true": rows.loc[mask, target].astype(int).to_numpy(),
            "probability": probability,
        }
    return output


def evaluate_policy(y_true, probability, threshold, scenario):
    predictions = (probability >= threshold).astype(int)
    metrics = safe_probability_metrics(y_true, probability, threshold)
    costs = classification_cost(
        y_true,
        predictions,
        false_positive_cost=scenario["false_positive_cost"],
        false_negative_cost=scenario["false_negative_cost"],
    )
    return {**metrics, **costs}


def save_figure(comparison, sensitivity, output_path):
    horizons = comparison["horizon_seconds"].to_numpy()
    x = np.arange(len(horizons))
    width = 0.34
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    colors = {"f1": "#64748B", "cost": "#DC2626"}

    axes[0].bar(
        x - width / 2,
        comparison["f1_predicted_fill_rate"],
        width,
        color=colors["f1"],
        label="Original F1 threshold",
    )
    axes[0].bar(
        x + width / 2,
        comparison["cost_predicted_fill_rate"],
        width,
        color=colors["cost"],
        label="3:1 cost threshold",
    )
    axes[0].axhline(SELECTIVITY_CEILING, color="#111827", linestyle="--", linewidth=1)
    axes[0].set_title("Test predicted positive rate", weight="bold", fontsize=11)
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(
        x - width / 2,
        comparison["f1_cost_per_order"],
        width,
        color=colors["f1"],
    )
    axes[1].bar(
        x + width / 2,
        comparison["cost_cost_per_order"],
        width,
        color=colors["cost"],
    )
    axes[1].set_title("Test normalized 3:1 error cost", weight="bold", fontsize=11)
    axes[1].set_ylim(bottom=0)

    for scenario_name, group in sensitivity.groupby("scenario"):
        axes[2].plot(
            group["horizon_seconds"],
            group["predicted_fill_rate"],
            marker="o",
            linewidth=2,
            label=scenario_name.replace("_", " "),
        )
    axes[2].set_title("Test sensitivity by cost scenario", weight="bold", fontsize=11)
    axes[2].set_ylim(0, 1)
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes[:2]:
        axis.set_xticks(x, [f"{h}s" for h in horizons])
    axes[2].set_xticks(HORIZONS)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Rate or normalized cost")
    figure.suptitle(
        "Decision-aware thresholds replace the near-universal 60-second policy",
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
    orders, policy = load_inputs(args.excel, args.policy)
    committed_predictions = pd.read_csv(
        args.results_dir / "final_test_predictions.csv"
    )
    result_rows = []
    comparison_rows = []
    reproduction_audit = []

    for horizon in HORIZONS:
        horizon_policy = policy["horizons"][str(horizon)]
        probabilities = reconstruct_probabilities(
            orders, horizon_policy, horizon, args.random_seed
        )
        committed = committed_predictions.loc[
            committed_predictions["horizon_seconds"].eq(horizon)
        ].sort_values("order_id")
        reconstructed = pd.DataFrame(
            {
                "order_id": probabilities["test"]["order_id"],
                "probability": probabilities["test"]["probability"],
            }
        ).sort_values("order_id")
        if committed["order_id"].tolist() != reconstructed["order_id"].tolist():
            raise ValueError(f"Committed test order IDs do not match at {horizon}s.")
        maximum_delta = float(
            np.max(
                np.abs(
                    committed["calibrated_probability"].to_numpy()
                    - reconstructed["probability"].to_numpy()
                )
            )
        )
        if maximum_delta > 1e-10:
            raise ValueError(
                f"Reconstructed {horizon}s probabilities differ by {maximum_delta}."
            )
        reproduction_audit.append(
            {"horizon_seconds": horizon, "maximum_probability_delta": maximum_delta}
        )

        selected_thresholds = {}
        for scenario_name, scenario in SCENARIOS.items():
            threshold = select_cost_threshold(
                probabilities["validation"]["y_true"],
                probabilities["validation"]["probability"],
                false_positive_cost=scenario["false_positive_cost"],
                false_negative_cost=scenario["false_negative_cost"],
                minimum_positive_rate=VALIDATION_POSITIVE_FLOOR,
                maximum_positive_rate=SELECTIVITY_CEILING,
            )
            selected_thresholds[scenario_name] = threshold
            for split in ("validation", "test"):
                evaluated = evaluate_policy(
                    probabilities[split]["y_true"],
                    probabilities[split]["probability"],
                    threshold,
                    scenario,
                )
                result_rows.append(
                    {
                        "horizon_seconds": horizon,
                        "scenario": scenario_name,
                        "split": split,
                        "false_positive_cost": scenario["false_positive_cost"],
                        "false_negative_cost": scenario["false_negative_cost"],
                        "selected_threshold": threshold,
                        **evaluated,
                    }
                )

        primary = SCENARIOS[PRIMARY_SCENARIO]
        primary_test = evaluate_policy(
            probabilities["test"]["y_true"],
            probabilities["test"]["probability"],
            selected_thresholds[PRIMARY_SCENARIO],
            primary,
        )
        f1_test = evaluate_policy(
            probabilities["test"]["y_true"],
            probabilities["test"]["probability"],
            float(horizon_policy["selected_threshold"]),
            primary,
        )
        comparison_rows.append(
            {
                "horizon_seconds": horizon,
                "f1_threshold": float(horizon_policy["selected_threshold"]),
                "cost_threshold": selected_thresholds[PRIMARY_SCENARIO],
                **{f"f1_{key}": value for key, value in f1_test.items()},
                **{f"cost_{key}": value for key, value in primary_test.items()},
            }
        )

    results = pd.DataFrame(result_rows).sort_values(
        ["scenario", "horizon_seconds", "split"]
    )
    comparison = pd.DataFrame(comparison_rows)
    primary_test = results.loc[
        results["scenario"].eq(PRIMARY_SCENARIO) & results["split"].eq("test")
    ].copy()
    selectivity_pass = primary_test["predicted_fill_rate"].gt(0) & primary_test[
        "predicted_fill_rate"
    ].le(SELECTIVITY_CEILING)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_dir / "decision_threshold_results.csv", index=False)
    comparison.to_csv(
        args.results_dir / "decision_threshold_comparison.csv", index=False
    )
    payload = {
        "status": "complete",
        "random_seed": args.random_seed,
        "objective": {
            "primary_scenario": PRIMARY_SCENARIO,
            "false_positive_definition": "predict fill when the order does not fill within the horizon; proxy for waiting and opportunity cost",
            "false_negative_definition": "predict no fill when the order fills within the horizon; proxy for missed passive execution",
            "scenarios": SCENARIOS,
            "threshold_rule": "minimize validation cost per eligible order subject to a 1%–80% validation positive-rate guardrail; break ties toward the calibrated-probability Bayes threshold",
            "economic_boundary": "costs are normalized sensitivity weights, not currency P&L or measured adverse selection",
        },
        "selectivity_guardrail": {
            "minimum_validation_positive_rate": VALIDATION_POSITIVE_FLOOR,
            "maximum_test_positive_rate": SELECTIVITY_CEILING,
            "test_policy_must_take_at_least_one_positive_action": True,
        },
        "acceptance": {
            "validation_only_threshold_selection": True,
            "f1_is_not_selection_objective": True,
            "all_primary_test_policies_selective": bool(selectivity_pass.all()),
            "passed": bool(selectivity_pass.all()),
        },
        "primary_test_results": primary_test.to_dict("records"),
        "f1_comparison": comparison_rows,
        "reproduction_audit": reproduction_audit,
        "limitations": [
            "Normalized error costs are illustrative and are not measured implementation shortfall.",
            "The workbook does not contain post-decision price paths needed to estimate adverse selection directly.",
            "Thresholds are evaluated on the same synthetic final test period used by the original study.",
        ],
    }
    (args.results_dir / "decision_threshold_summary.json").write_text(
        json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8"
    )
    save_figure(
        comparison,
        results.loc[results["split"].eq("test")],
        args.figure_dir / "decision_threshold_tradeoffs.png",
    )
    print(
        "Decision-threshold evaluation complete: "
        f"selectivity acceptance passed={bool(selectivity_pass.all())}."
    )


if __name__ == "__main__":
    main()
