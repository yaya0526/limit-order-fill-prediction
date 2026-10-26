"""Fast, raw-data-free validation for the published research repository."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = (
    "README.md",
    "RESULTS.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "requirements.txt",
    "data/README.md",
    "docs/feature_reference.md",
    "docs/limitations_and_roadmap.md",
    "docs/rolling_origin_robustness.md",
    "docs/releases/v0.1.0.md",
    "scripts/demo_results.py",
    "scripts/run_rolling_origin.py",
    "results/final_test_metrics.csv",
    "results/final_test_summary.json",
    "results/rolling_origin_predictions.csv",
    "results/rolling_origin_summary.csv",
    "results/rolling_origin_summary.json",
    "results/rolling_origin_window_metrics.csv",
    "results/selected_model_policy.json",
    "reports/figures/rolling_origin_robustness.png",
)

EXPECTED_NOTEBOOKS = (
    "01_data_audit.ipynb",
    "02_exploratory_analysis.ipynb",
    "03_history_features.ipynb",
    "04_model_ablation.ipynb",
    "05_model_selection_calibration.ipynb",
    "06_final_test_evaluation.ipynb",
    "07_model_explainability.ipynb",
    "08_brief_required_explanations.ipynb",
)

METRIC_COLUMNS = {
    "horizon_seconds",
    "candidate",
    "feature_set",
    "model",
    "calibration",
    "selected_threshold",
    "eligible_rows",
    "true_fill_rate",
    "predicted_fill_rate",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "ece_10",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
}

TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

PERSONAL_PATH_PATTERNS = (
    re.compile(r"/Users/[^/\s]+/"),
    re.compile(r"/home/[^/\s]+/"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
)

IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "env",
    "node_modules",
    "venv",
}


def is_ignored_path(path: Path) -> bool:
    return any(part in IGNORED_DIRECTORY_NAMES for part in path.relative_to(ROOT).parts)


def validate_required_paths() -> list[str]:
    return [f"missing required path: {path}" for path in REQUIRED_PATHS if not (ROOT / path).exists()]


def validate_notebooks() -> list[str]:
    errors: list[str] = []
    notebook_dir = ROOT / "notebooks"
    actual = tuple(sorted(path.name for path in notebook_dir.glob("*.ipynb")))
    if actual != EXPECTED_NOTEBOOKS:
        errors.append(f"unexpected notebook sequence: {actual}")
    for name in actual:
        path = notebook_dir / name
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid notebook JSON {path.relative_to(ROOT)}: {exc}")
            continue
        if notebook.get("nbformat") != 4:
            errors.append(f"{path.relative_to(ROOT)} must use notebook format 4")
        if not isinstance(notebook.get("cells"), list):
            errors.append(f"{path.relative_to(ROOT)} has no cell list")
    return errors


def validate_metrics() -> list[str]:
    errors: list[str] = []
    path = ROOT / "results" / "final_test_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    missing = METRIC_COLUMNS - fields
    if missing:
        errors.append(f"final_test_metrics.csv missing columns: {sorted(missing)}")
        return errors
    horizons = [int(row["horizon_seconds"]) for row in rows]
    if horizons != [5, 30, 60]:
        errors.append(f"expected horizons [5, 30, 60], found {horizons}")
    for row in rows:
        horizon = row["horizon_seconds"]
        eligible = int(row["eligible_rows"])
        confusion_total = sum(int(row[key]) for key in ("tn", "fp", "fn", "tp"))
        if confusion_total != eligible:
            errors.append(f"{horizon}s confusion matrix totals {confusion_total}, not {eligible}")
        for key in (
            "selected_threshold",
            "true_fill_rate",
            "predicted_fill_rate",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "ece_10",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ):
            value = float(row[key])
            if not 0 <= value <= 1:
                errors.append(f"{horizon}s {key} is outside [0, 1]: {value}")
    return errors


def validate_json_artifacts() -> list[str]:
    errors: list[str] = []
    for relative in ("results/final_test_summary.json", "results/selected_model_policy.json"):
        path = ROOT / relative
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON {relative}: {exc}")
            continue
        if not isinstance(payload, dict) or not payload:
            errors.append(f"{relative} must contain a non-empty JSON object")
    return errors


def validate_rolling_origin() -> list[str]:
    errors: list[str] = []
    path = ROOT / "results" / "rolling_origin_window_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected_pairs = [(horizon, window) for horizon in (5, 30, 60) for window in ("W1", "W2", "W3")]
    actual_pairs = [(int(row["horizon_seconds"]), row["window_id"]) for row in rows]
    if actual_pairs != expected_pairs:
        errors.append(f"unexpected rolling-origin horizon/window sequence: {actual_pairs}")
        return errors

    window_ranges: dict[str, tuple[str, str]] = {}
    for row in rows:
        eligible = int(row["eligible_rows"])
        confusion_total = sum(int(row[key]) for key in ("tn", "fp", "fn", "tp"))
        if confusion_total != eligible:
            errors.append(
                f"rolling {row['horizon_seconds']}s {row['window_id']} confusion total "
                f"{confusion_total}, not {eligible}"
            )
        for key in (
            "selected_threshold",
            "true_fill_rate",
            "predicted_fill_rate",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "ece_10",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ):
            value = float(row[key])
            if not 0 <= value <= 1:
                errors.append(
                    f"rolling {row['horizon_seconds']}s {row['window_id']} "
                    f"{key} is outside [0, 1]: {value}"
                )
        observed_range = (row["test_start"], row["test_end"])
        existing_range = window_ranges.setdefault(row["window_id"], observed_range)
        if existing_range != observed_range:
            errors.append(f"inconsistent dates for rolling window {row['window_id']}")

    ordered_ranges = sorted(window_ranges.values())
    for previous, current in zip(ordered_ranges, ordered_ranges[1:]):
        if previous[1] >= current[0]:
            errors.append(f"overlapping rolling test windows: {previous} and {current}")

    summary_path = ROOT / "results" / "rolling_origin_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON results/rolling_origin_summary.json: {exc}")
        return errors
    acceptance = summary.get("acceptance", {})
    if acceptance.get("observed_non_overlapping_test_windows") != 3:
        errors.append("rolling-origin summary must report three non-overlapping test windows")
    if acceptance.get("dispersion_reported_with_mean") is not True:
        errors.append("rolling-origin summary must report dispersion with mean performance")
    if acceptance.get("passed") is not True:
        errors.append("rolling-origin acceptance record is not passing")
    if summary.get("design", {}).get("original_final_test_untouched") is not True:
        errors.append("rolling-origin summary must keep the original final test untouched")
    return errors


def validate_markdown_links() -> list[str]:
    errors: list[str] = []
    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for path in sorted(ROOT.rglob("*.md")):
        if is_ignored_path(path):
            continue
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        for raw_target in link_pattern.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / unquote(target)).resolve()
            if ROOT not in resolved.parents and resolved != ROOT:
                errors.append(f"{relative} link escapes repository: {raw_target}")
            elif not resolved.exists():
                errors.append(f"{relative} has broken local link: {raw_target}")
    return errors


def validate_no_personal_paths() -> list[str]:
    errors: list[str] = []
    this_file = Path(__file__).resolve()
    for path in ROOT.rglob("*"):
        if not path.is_file() or is_ignored_path(path):
            continue
        if path.resolve() == this_file:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"Makefile"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in PERSONAL_PATH_PATTERNS):
            errors.append(f"personal absolute path found in {path.relative_to(ROOT)}")
    return errors


def validate_repository() -> list[str]:
    errors: list[str] = []
    for check in (
        validate_required_paths,
        validate_notebooks,
        validate_metrics,
        validate_json_artifacts,
        validate_rolling_origin,
        validate_markdown_links,
        validate_no_personal_paths,
    ):
        errors.extend(check())
    return errors


def main() -> int:
    errors = validate_repository()
    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Repository validation passed: structure, artifacts, links, and privacy checks are clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
