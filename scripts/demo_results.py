"""Render a raw-data-free summary of the committed held-out results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_PATH = ROOT / "results" / "final_test_metrics.csv"

REQUIRED_COLUMNS = {
    "horizon_seconds",
    "feature_set",
    "model",
    "eligible_rows",
    "true_fill_rate",
    "predicted_fill_rate",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "ece_10",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
}

def load_results(path: Path = DEFAULT_METRICS_PATH) -> list[dict[str, str]]:
    """Load and validate the compact published metric table."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"missing required metric columns: {sorted(missing)}")
    if not rows:
        raise ValueError("the metric table is empty")

    horizons = [int(row["horizon_seconds"]) for row in rows]
    if horizons != [5, 30, 60]:
        raise ValueError(f"expected horizons [5, 30, 60], found {horizons}")

    for row in rows:
        eligible = int(row["eligible_rows"])
        confusion_total = sum(int(row[key]) for key in ("tn", "fp", "fn", "tp"))
        if confusion_total != eligible:
            raise ValueError(
                f"{row['horizon_seconds']}s confusion counts total "
                f"{confusion_total}, not {eligible}"
            )
        for key in (
            "true_fill_rate",
            "predicted_fill_rate",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "ece_10",
            "f1",
        ):
            value = float(row[key])
            if not 0 <= value <= 1:
                raise ValueError(
                    f"{row['horizon_seconds']}s {key} is outside [0, 1]: {value}"
                )
    return rows


def compact_results(rows: Sequence[dict[str, str]]) -> list[dict[str, object]]:
    """Return the reviewer-facing fields with stable numeric types."""

    compact: list[dict[str, object]] = []
    for row in rows:
        compact.append(
            {
                "horizon_seconds": int(row["horizon_seconds"]),
                "model": row["model"],
                "feature_set": row["feature_set"],
                "eligible_rows": int(row["eligible_rows"]),
                "true_fill_rate": float(row["true_fill_rate"]),
                "predicted_fill_rate": float(row["predicted_fill_rate"]),
                "roc_auc": float(row["roc_auc"]),
                "pr_auc": float(row["pr_auc"]),
                "brier_score": float(row["brier_score"]),
                "ece_10": float(row["ece_10"]),
                "f1": float(row["f1"]),
            }
        )
    return compact


def build_takeaways(rows: Sequence[dict[str, str]]) -> list[str]:
    """Derive concise findings from the committed metrics."""

    strongest = max(rows, key=lambda row: float(row["pr_auc"]))
    least_selective = max(rows, key=lambda row: float(row["predicted_fill_rate"]))
    shortest = min(rows, key=lambda row: int(row["horizon_seconds"]))
    return [
        (
            f"The strongest ranking result is {strongest['horizon_seconds']}s "
            f"(PR-AUC {float(strongest['pr_auc']):.3f}; observed fill rate "
            f"{float(strongest['true_fill_rate']):.1%})."
        ),
        (
            f"The {least_selective['horizon_seconds']}s operating threshold is weakly "
            f"selective: {float(least_selective['predicted_fill_rate']):.1%} of eligible "
            f"orders are predicted to fill versus "
            f"{float(least_selective['true_fill_rate']):.1%} observed."
        ),
        (
            f"The shortest horizon remains difficult: {shortest['horizon_seconds']}s "
            f"ROC-AUC is {float(shortest['roc_auc']):.3f} and PR-AUC is "
            f"{float(shortest['pr_auc']):.3f}."
        ),
    ]


DISPLAY_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "B_order_lob": "Order + LOB",
    "C_order_lob_flow": "Order + LOB + flow",
}


def _display_name(value: str) -> str:
    return DISPLAY_NAMES.get(value, value.replace("_", " ").capitalize())


def render_text(rows: Sequence[dict[str, str]], source: Path) -> str:
    """Render a terminal-friendly table and the main interpretation caveats."""

    headers = ("Horizon", "Model", "Features", "N", "Actual", "Predicted", "ROC", "PR", "Brier", "ECE", "F1")
    table_rows = []
    for row in rows:
        table_rows.append(
            (
                f"{int(row['horizon_seconds'])}s",
                _display_name(row["model"]),
                _display_name(row["feature_set"]),
                f"{int(row['eligible_rows']):,}",
                f"{float(row['true_fill_rate']):.1%}",
                f"{float(row['predicted_fill_rate']):.1%}",
                f"{float(row['roc_auc']):.3f}",
                f"{float(row['pr_auc']):.3f}",
                f"{float(row['brier_score']):.3f}",
                f"{float(row['ece_10']):.3f}",
                f"{float(row['f1']):.3f}",
            )
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]

    def format_row(values: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    try:
        displayed_source = source.resolve().relative_to(ROOT)
    except ValueError:
        displayed_source = source.resolve()
    lines = [
        "Limit Order Fill Probability — committed held-out results",
        f"Source: {displayed_source} (no raw data required)",
        "",
        format_row(headers),
        format_row(tuple("-" * width for width in widths)),
        *(format_row(row) for row in table_rows),
        "",
        "Key takeaways",
        *(f"- {takeaway}" for takeaway in build_takeaways(rows)),
        "",
        "Boundary: these results use deterministic synthetic teaching data; they are",
        "not evidence of live AAPL execution quality or trading profitability.",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display the committed final metrics without the restricted raw data."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="metric CSV to display (default: results/final_test_metrics.csv)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the compact reviewer-facing fields as JSON",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = load_results(args.metrics)
    except (OSError, ValueError) as exc:
        print(f"Unable to load published results: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(compact_results(rows), indent=2))
    else:
        print(render_text(rows, args.metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
