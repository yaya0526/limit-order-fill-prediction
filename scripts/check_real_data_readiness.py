"""Check an authorized normalized event-data package before R5 modelling."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_data_readiness import file_sha256, validate_event_rows, validate_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate provenance and normalized L3 events before R5 analysis."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Emit a JSON report.")
    return parser.parse_args(argv)


def validate_package(manifest_path: Path, event_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_manifest(manifest)

    expected_digest = None
    manifest_root = manifest_path.resolve().parent
    for entry in manifest.get("files", []):
        if not isinstance(entry, dict) or entry.get("kind") != "events":
            continue
        declared = (manifest_root / str(entry.get("path", ""))).resolve()
        if declared == event_path.resolve():
            expected_digest = str(entry.get("sha256", ""))
            break
    if expected_digest is None:
        errors.append("events path is not declared as an events file in the manifest")
    elif event_path.is_file():
        observed_digest = file_sha256(event_path)
        if observed_digest != expected_digest:
            errors.append("events file SHA-256 does not match the manifest")
    else:
        errors.append(f"events file not found: {event_path}")

    event_summary = {}
    if event_path.is_file():
        with event_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            row_errors, event_summary = validate_event_rows(
                reader.fieldnames or [], reader, manifest
            )
        errors.extend(row_errors)
    return {
        "status": "ready" if not errors else "not_ready",
        "manifest": str(manifest_path),
        "events": str(event_path),
        "errors": errors,
        "event_summary": event_summary,
        "gates": {
            "provenance_and_entitlement": not any(
                "manifest" in error or "entitlement" in error for error in errors
            ),
            "file_integrity": not any("SHA-256" in error for error in errors),
            "event_schema_and_ordering": not any(
                error.startswith("event file") or error.startswith("row ")
                for error in errors
            ),
        },
    }


def main(argv=None):
    args = parse_args(argv)
    try:
        report = validate_package(args.manifest, args.events)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"Unable to validate real-data package: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
    elif report["status"] == "ready":
        summary = report["event_summary"]
        print(
            "Real-data intake gates passed: "
            f"{summary.get('row_count', 0):,} events, "
            f"{len(summary.get('trading_days', []))} days, "
            f"{len(summary.get('symbols', []))} symbols."
        )
    else:
        print("Real-data intake is not ready:")
        for error in report["errors"]:
            print(f"- {error}")
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
