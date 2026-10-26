"""Run R4 transfer stress tests on an independently configured simulator."""

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
from src.market_simulation import (
    HORIZONS,
    SCENARIOS,
    scenario_manifest,
    scenario_manifest_hash,
    simulate_scenario,
    transfer_status,
)
from src.modeling import FEATURE_SETS, add_oriented_static_features
from src.rolling_origin import build_expanding_calibration_folds


PRIMARY_DECISION_SCENARIO = "precision_first"
MINIMUM_PR_AUC_LIFT = 0.02


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate frozen fill policies on independent market scenarios."
    )
    parser.add_argument("--excel", type=Path, required=True, help="Source Excel workbook.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "results" / "selected_model_policy.json",
        help="Frozen model policy JSON.",
    )
    parser.add_argument(
        "--decision-results",
        type=Path,
        default=ROOT / "results" / "decision_threshold_results.csv",
        help="Committed R3 threshold results.",
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
    parser.add_argument("--rows-per-scenario", type=int, default=4_000)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--simulation-seed", type=int, default=20261026)
    return parser.parse_args()


def load_training_inputs(excel_path, policy_path):
    if not excel_path.is_file():
        raise FileNotFoundError(f"Workbook not found: {excel_path}")
    orders = pd.read_excel(excel_path, sheet_name="Orders")
    if len(orders) != 10_000 or not orders["order_id"].is_unique:
        raise ValueError("Expected 10,000 rows with unique order_id values.")
    orders["trading_day_parsed"] = pd.to_datetime(orders["trading_day"])
    orders["data_split_normalized"] = orders["data_split"].astype(str).str.lower()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    return add_oriented_static_features(orders), policy


def load_frozen_thresholds(path):
    results = pd.read_csv(path)
    rows = results.loc[
        results["scenario"].eq(PRIMARY_DECISION_SCENARIO)
        & results["split"].eq("validation")
    ]
    thresholds = {
        int(row.horizon_seconds): float(row.selected_threshold)
        for row in rows.itertuples()
    }
    if set(thresholds) != set(HORIZONS):
        raise ValueError("Expected one frozen primary R3 threshold per horizon.")
    return thresholds


def fit_frozen_probability_model(orders, horizon_policy, horizon, random_seed):
    """Reconstruct the training-only estimator and OOF sigmoid calibrator."""
    feature_set = horizon_policy["feature_set"]
    model_name = horizon_policy["model"]
    parameters = horizon_policy["parameters"]
    features = FEATURE_SETS[feature_set]
    eligibility = f"eligible_{horizon}s"
    target = f"target_fill_{horizon}s"
    training = orders.loc[orders["data_split_normalized"].eq("train")].copy()
    training_days = sorted(training["trading_day_parsed"].unique())

    oof_probability = []
    oof_target = []
    for fit_days, score_days in build_expanding_calibration_folds(training_days):
        fit_mask = training["trading_day_parsed"].isin(fit_days) & training[
            eligibility
        ].eq(1)
        score_mask = training["trading_day_parsed"].isin(score_days) & training[
            eligibility
        ].eq(1)
        estimator = build_estimator(
            feature_set, model_name, parameters, random_seed=random_seed
        )
        estimator.fit(
            training.loc[fit_mask, features],
            training.loc[fit_mask, target].astype(int),
        )
        oof_probability.append(
            estimator.predict_proba(training.loc[score_mask, features])[:, 1]
        )
        oof_target.append(training.loc[score_mask, target].astype(int).to_numpy())

    calibrator = fit_sigmoid_calibrator(
        np.concatenate(oof_probability), np.concatenate(oof_target)
    )
    training_mask = training[eligibility].eq(1)
    estimator = build_estimator(
        feature_set, model_name, parameters, random_seed=random_seed
    )
    estimator.fit(
        training.loc[training_mask, features],
        training.loc[training_mask, target].astype(int),
    )
    return estimator, calibrator, features


def evaluate_transfer(y_true, probability, threshold):
    metrics = safe_probability_metrics(y_true, probability, threshold)
    prevalence = float(np.mean(y_true))
    null_brier = prevalence * (1 - prevalence)
    pr_auc_lift = float(metrics["pr_auc"] - prevalence)
    brier_skill = (
        float(1 - metrics["brier_score"] / null_brier) if null_brier > 0 else np.nan
    )
    status = transfer_status(
        pr_auc_lift, brier_skill, minimum_pr_lift=MINIMUM_PR_AUC_LIFT
    )
    return {
        **metrics,
        "prevalence_baseline_pr_auc": prevalence,
        "pr_auc_lift": pr_auc_lift,
        "null_brier_score": null_brier,
        "brier_skill_score": brier_skill,
        "transfer_status": status,
    }


def summarize_scenarios(metrics):
    summaries = []
    for scenario, group in metrics.groupby("scenario", sort=False):
        transfer_horizons = int(group["transfer_status"].ne("failure").sum())
        if transfer_horizons >= 2:
            status = "successful_transfer_regime"
        elif transfer_horizons == 0:
            status = "failure_regime"
        else:
            status = "mixed_regime"
        summaries.append(
            {
                "scenario": scenario,
                "scenario_status": status,
                "transfer_horizons": transfer_horizons,
                "failed_horizons": int(len(group) - transfer_horizons),
                "mean_pr_auc_lift": float(group["pr_auc_lift"].mean()),
                "mean_brier_skill_score": float(group["brier_skill_score"].mean()),
            }
        )
    return summaries


def save_figure(metrics, output_path):
    scenarios = [scenario.name for scenario in SCENARIOS]
    labels = [name.replace("_", "\n") for name in scenarios]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharex=True)
    colors = {5: "#2563EB", 30: "#D97706", 60: "#059669"}
    x = np.arange(len(scenarios))
    width = 0.24
    columns = (
        ("pr_auc_lift", "PR-AUC lift over prevalence", (-0.05, 0.75)),
        ("brier_skill_score", "Brier skill vs prevalence", (-60, 0.8)),
        ("predicted_fill_rate", "Frozen-policy positive rate", (0, 0.06)),
    )
    for axis, (column, title, limits) in zip(axes, columns):
        for offset, horizon in enumerate(HORIZONS):
            group = metrics.loc[metrics["horizon_seconds"].eq(horizon)].set_index(
                "scenario"
            )
            values = [float(group.loc[name, column]) for name in scenarios]
            axis.bar(
                x + (offset - 1) * width,
                values,
                width,
                label=f"{horizon}s",
                color=colors[horizon],
            )
        axis.axhline(0, color="#111827", linewidth=0.8)
        axis.set_title(title, weight="bold", fontsize=10.5)
        axis.set_ylim(*limits)
        axis.set_xticks(x, labels, fontsize=8)
        axis.grid(axis="y", alpha=0.23)
    axes[0].axhline(
        MINIMUM_PR_AUC_LIFT, color="#DC2626", linestyle="--", linewidth=1
    )
    axes[1].set_yscale("symlog", linthresh=0.25, linscale=0.8)
    axes[1].set_title("Brier skill (symmetric log scale)", weight="bold", fontsize=10.5)
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    figure.suptitle(
        "Frozen models transfer unevenly across independent market mechanisms",
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
    orders, policy = load_training_inputs(args.excel, args.policy)
    thresholds = load_frozen_thresholds(args.decision_results)
    simulated = {
        config.name: add_oriented_static_features(
            simulate_scenario(
                config,
                rows=args.rows_per_scenario,
                random_seed=args.simulation_seed,
            )
        )
        for config in SCENARIOS
    }

    rows = []
    for horizon in HORIZONS:
        horizon_policy = policy["horizons"][str(horizon)]
        estimator, calibrator, features = fit_frozen_probability_model(
            orders, horizon_policy, horizon, args.model_seed
        )
        for config in SCENARIOS:
            frame = simulated[config.name]
            raw_probability = estimator.predict_proba(frame[features])[:, 1]
            probability = apply_sigmoid_calibrator(calibrator, raw_probability)
            y_true = frame[f"target_fill_{horizon}s"].to_numpy(dtype=int)
            evaluated = evaluate_transfer(y_true, probability, thresholds[horizon])
            rows.append(
                {
                    "scenario": config.name,
                    "mechanism": config.mechanism,
                    "horizon_seconds": horizon,
                    "frozen_candidate": horizon_policy["candidate"],
                    "frozen_threshold": thresholds[horizon],
                    **evaluated,
                }
            )

    metrics = pd.DataFrame(rows).sort_values(["scenario", "horizon_seconds"])
    scenario_summaries = summarize_scenarios(metrics)
    successful = sum(
        item["scenario_status"] == "successful_transfer_regime"
        for item in scenario_summaries
    )
    failures = sum(
        item["scenario_status"] == "failure_regime" for item in scenario_summaries
    )
    acceptance = {
        "scenario_parameters_declared_before_scoring": True,
        "source_test_set_not_used": True,
        "simulator_does_not_read_source_workbook": True,
        "no_scenario_specific_model_or_threshold_retuning": True,
        "successful_transfer_regimes": successful,
        "failure_regimes": failures,
        "both_success_and_failure_documented": bool(successful and failures),
        "passed": bool(successful and failures),
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.results_dir / "independent_simulation_metrics.csv", index=False)
    payload = {
        "status": "complete" if acceptance["passed"] else "completed_with_failed_acceptance",
        "design": {
            "rows_per_scenario": args.rows_per_scenario,
            "model_random_seed": args.model_seed,
            "simulation_random_seed": args.simulation_seed,
            "scenario_manifest_sha256": scenario_manifest_hash(),
            "transfer_rule": (
                "PR-AUC must exceed scenario prevalence by more than 0.02; positive "
                "Brier skill additionally indicates calibration transfer"
            ),
            "frozen_decision_scenario": PRIMARY_DECISION_SCENARIO,
            "generator_boundary": (
                "Scenario parameters are code constants and are not estimated from the "
                "source workbook; source training rows are used only to reconstruct the "
                "already frozen models and calibrators."
            ),
        },
        "scenario_manifest": scenario_manifest(),
        "scenario_summaries": scenario_summaries,
        "acceptance": acceptance,
        "limitations": [
            "The independent generator is a transparent queue-depletion benchmark, not an exchange emulator.",
            "It omits message-level latency, partial-fill paths, amendments, and venue-specific priority rules.",
            "A scenario-level success is evidence of limited synthetic transfer, not evidence about live AAPL.",
        ],
    }
    (args.results_dir / "independent_simulation_summary.json").write_text(
        json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8"
    )
    save_figure(metrics, args.figure_dir / "independent_simulation_transfer.png")
    print(
        "Independent simulation complete: "
        f"successful regimes={successful}, failure regimes={failures}, "
        f"acceptance passed={acceptance['passed']}."
    )
    if not acceptance["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
