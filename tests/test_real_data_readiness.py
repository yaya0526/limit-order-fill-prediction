import csv
import hashlib
import json

from scripts.check_real_data_readiness import validate_package
from src.real_data_readiness import (
    REQUIRED_EVENT_COLUMNS,
    validate_event_rows,
    validate_manifest,
)


def valid_manifest(digest="a" * 64):
    return {
        "dataset_id": "licensed-sample-001",
        "provider": "authorized-provider",
        "license_reference": "internal-entitlement-001",
        "entitlement_confirmed": True,
        "venue": "TESTX",
        "symbols": ["AAPL"],
        "timezone": "America/New_York",
        "timestamp_unit": "nanoseconds",
        "tick_size": 0.01,
        "schema_version": "1.0",
        "date_range": {"start": "2026-01-05", "end": "2026-01-06"},
        "files": [
            {"path": "events.csv", "sha256": digest, "kind": "events"}
        ],
    }


def valid_rows():
    base = {
        "dataset_id": "licensed-sample-001",
        "venue": "TESTX",
        "symbol": "AAPL",
        "trading_day": "2026-01-05",
        "side": "buy",
        "price_ticks": "14000",
        "quantity": "100",
    }
    return [
        {
            **base,
            "event_timestamp_ns": "1000",
            "sequence_number": "1",
            "message_type": "add",
            "order_id": "O1",
        },
        {
            **base,
            "event_timestamp_ns": "1100",
            "sequence_number": "2",
            "message_type": "cancel",
            "order_id": "O1",
        },
        {
            **base,
            "event_timestamp_ns": "1200",
            "sequence_number": "3",
            "message_type": "execute",
            "order_id": "O2",
        },
    ]


def test_valid_manifest_passes_provenance_gate():
    assert validate_manifest(valid_manifest()) == []


def test_manifest_rejects_missing_entitlement_and_placeholder_provenance():
    manifest = valid_manifest()
    manifest["entitlement_confirmed"] = False
    manifest["license_reference"] = "TBD"
    errors = validate_manifest(manifest)
    assert any("entitlement_confirmed" in error for error in errors)
    assert any("license_reference" in error for error in errors)


def test_normalized_event_stream_passes_ordering_and_control_totals():
    rows = valid_rows()
    errors, summary = validate_event_rows(
        REQUIRED_EVENT_COLUMNS, rows, valid_manifest()
    )
    assert errors == []
    assert summary["row_count"] == 3
    assert summary["message_counts"] == {"add": 1, "cancel": 1, "execute": 1}


def test_event_stream_rejects_leakage_and_nonincreasing_sequence():
    rows = valid_rows()
    rows[2]["sequence_number"] = "2"
    errors, _ = validate_event_rows(
        REQUIRED_EVENT_COLUMNS | {"target_fill_30s"}, rows, valid_manifest()
    )
    assert any("future/label" in error for error in errors)
    assert any("sequence numbers" in error for error in errors)


def test_end_to_end_package_verifies_declared_sha256(tmp_path):
    events_path = tmp_path / "events.csv"
    rows = valid_rows()
    with events_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_EVENT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(events_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(valid_manifest(digest)), encoding="utf-8"
    )

    report = validate_package(manifest_path, events_path)
    assert report["status"] == "ready"
    assert report["errors"] == []
