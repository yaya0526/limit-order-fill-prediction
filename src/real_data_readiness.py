"""Validation helpers for the R5 licensed event-data intake gate."""

from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import re
from pathlib import Path
from typing import Iterable, Mapping


REQUIRED_MANIFEST_FIELDS = {
    "dataset_id",
    "provider",
    "license_reference",
    "entitlement_confirmed",
    "venue",
    "symbols",
    "timezone",
    "timestamp_unit",
    "tick_size",
    "schema_version",
    "date_range",
    "files",
}

REQUIRED_EVENT_COLUMNS = {
    "dataset_id",
    "venue",
    "symbol",
    "trading_day",
    "event_timestamp_ns",
    "sequence_number",
    "message_type",
    "side",
    "order_id",
    "price_ticks",
    "quantity",
}

ALLOWED_MESSAGE_TYPES = {"add", "modify", "cancel", "execute"}
ALLOWED_SIDES = {"buy", "sell"}
BANNED_LEAKAGE_COLUMNS = {
    "target_fill_5s",
    "target_fill_30s",
    "target_fill_60s",
    "future_fill",
    "future_price",
    "realized_shortfall",
}
PLACEHOLDER_VALUES = {"", "unknown", "tbd", "todo", "none", "n/a"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _missing_or_placeholder(value) -> bool:
    return value is None or str(value).strip().lower() in PLACEHOLDER_VALUES


def _parse_iso_date(value, field, errors):
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        errors.append(f"{field} must be an ISO date (YYYY-MM-DD)")
        return None


def validate_manifest(manifest: Mapping) -> list[str]:
    """Validate provenance, entitlement, time, and file-integrity metadata."""
    errors: list[str] = []
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        errors.append(f"manifest missing fields: {sorted(missing)}")
        return errors

    for field in (
        "dataset_id",
        "provider",
        "license_reference",
        "venue",
        "timezone",
        "schema_version",
    ):
        if _missing_or_placeholder(manifest.get(field)):
            errors.append(f"manifest {field} must be explicit")
    if manifest.get("entitlement_confirmed") is not True:
        errors.append("manifest entitlement_confirmed must be true")
    if manifest.get("timestamp_unit") != "nanoseconds":
        errors.append("manifest timestamp_unit must be nanoseconds")
    try:
        if float(manifest.get("tick_size")) <= 0:
            errors.append("manifest tick_size must be positive")
    except (TypeError, ValueError):
        errors.append("manifest tick_size must be numeric")

    symbols = manifest.get("symbols")
    if not isinstance(symbols, list) or not symbols or any(
        _missing_or_placeholder(symbol) for symbol in symbols
    ):
        errors.append("manifest symbols must be a non-empty explicit list")

    date_range = manifest.get("date_range")
    if not isinstance(date_range, Mapping):
        errors.append("manifest date_range must contain start and end")
    else:
        start = _parse_iso_date(date_range.get("start"), "date_range.start", errors)
        end = _parse_iso_date(date_range.get("end"), "date_range.end", errors)
        if start and end and start > end:
            errors.append("manifest date_range.start must not exceed date_range.end")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        errors.append("manifest files must be a non-empty list")
    else:
        observed_paths = set()
        for index, entry in enumerate(files):
            if not isinstance(entry, Mapping):
                errors.append(f"manifest files[{index}] must be an object")
                continue
            path = str(entry.get("path", ""))
            digest = str(entry.get("sha256", "")).lower()
            kind = str(entry.get("kind", ""))
            if _missing_or_placeholder(path):
                errors.append(f"manifest files[{index}].path must be explicit")
            elif Path(path).is_absolute() or ".." in Path(path).parts:
                errors.append(f"manifest files[{index}].path must be relative and contained")
            elif path in observed_paths:
                errors.append(f"manifest contains duplicate file path: {path}")
            observed_paths.add(path)
            if not SHA256_PATTERN.fullmatch(digest):
                errors.append(f"manifest files[{index}].sha256 must be 64 lowercase hex characters")
            if kind not in {"events", "reference", "corporate_actions"}:
                errors.append(f"manifest files[{index}].kind is unsupported: {kind}")
        if not any(
            isinstance(entry, Mapping) and entry.get("kind") == "events"
            for entry in files
        ):
            errors.append("manifest must include at least one events file")
    return errors


def file_sha256(path: Path, chunk_size: int = 1_048_576) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_event_rows(
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, str]],
    manifest: Mapping,
) -> tuple[list[str], dict]:
    """Validate a normalized L3 event stream without loading it into memory."""
    errors: list[str] = []
    columns = set(fieldnames)
    missing = REQUIRED_EVENT_COLUMNS - columns
    if missing:
        return [f"event file missing columns: {sorted(missing)}"], {}
    leaked = BANNED_LEAKAGE_COLUMNS & columns
    if leaked:
        errors.append(f"event file contains future/label columns: {sorted(leaked)}")

    expected_dataset = str(manifest.get("dataset_id", ""))
    expected_venue = str(manifest.get("venue", ""))
    expected_symbols = set(map(str, manifest.get("symbols", [])))
    date_range = manifest.get("date_range", {})
    start = _parse_iso_date(date_range.get("start"), "date_range.start", errors)
    end = _parse_iso_date(date_range.get("end"), "date_range.end", errors)

    counts = Counter()
    row_count = 0
    observed_symbols = set()
    observed_days = set()
    last_by_stream: dict[tuple[str, str], tuple[int, int]] = {}
    limited_row_errors = 0

    def row_error(message):
        nonlocal limited_row_errors
        if limited_row_errors < 25:
            errors.append(message)
        limited_row_errors += 1

    for row_number, row in enumerate(rows, start=2):
        row_count += 1
        dataset_id = str(row.get("dataset_id", ""))
        venue = str(row.get("venue", ""))
        symbol = str(row.get("symbol", ""))
        trading_day_raw = str(row.get("trading_day", ""))
        message_type = str(row.get("message_type", "")).lower()
        side = str(row.get("side", "")).lower()
        order_id = str(row.get("order_id", ""))
        if dataset_id != expected_dataset:
            row_error(f"row {row_number}: dataset_id does not match manifest")
        if venue != expected_venue:
            row_error(f"row {row_number}: venue does not match manifest")
        if symbol not in expected_symbols:
            row_error(f"row {row_number}: symbol is outside manifest scope")
        if message_type not in ALLOWED_MESSAGE_TYPES:
            row_error(f"row {row_number}: unsupported message_type {message_type!r}")
        if side not in ALLOWED_SIDES:
            row_error(f"row {row_number}: side must be buy or sell")
        if _missing_or_placeholder(order_id):
            row_error(f"row {row_number}: order_id must be explicit")

        trading_day = _parse_iso_date(
            trading_day_raw, f"row {row_number} trading_day", errors
        )
        if trading_day and start and end and not start <= trading_day <= end:
            row_error(f"row {row_number}: trading_day is outside manifest date range")
        try:
            timestamp = int(row.get("event_timestamp_ns", ""))
            sequence = int(row.get("sequence_number", ""))
            price_ticks = int(row.get("price_ticks", ""))
            quantity = int(row.get("quantity", ""))
            if timestamp <= 0 or sequence < 0 or price_ticks <= 0 or quantity <= 0:
                raise ValueError
        except (TypeError, ValueError):
            row_error(
                f"row {row_number}: timestamp/sequence/price_ticks/quantity values are invalid"
            )
            continue

        stream = (symbol, trading_day_raw)
        previous = last_by_stream.get(stream)
        if previous is not None:
            previous_timestamp, previous_sequence = previous
            if timestamp < previous_timestamp:
                row_error(f"row {row_number}: event timestamps decrease within stream")
            if sequence <= previous_sequence:
                row_error(f"row {row_number}: sequence numbers are not strictly increasing")
        last_by_stream[stream] = (timestamp, sequence)
        counts[message_type] += 1
        observed_symbols.add(symbol)
        observed_days.add(trading_day_raw)

    if row_count == 0:
        errors.append("event file contains no data rows")
    for required_type in ("add", "cancel", "execute"):
        if counts[required_type] == 0:
            errors.append(f"event file contains no {required_type} messages")
    if limited_row_errors > 25:
        errors.append(f"{limited_row_errors - 25} additional row errors suppressed")

    summary = {
        "row_count": row_count,
        "message_counts": dict(sorted(counts.items())),
        "symbols": sorted(observed_symbols),
        "trading_days": sorted(observed_days),
        "stream_count": len(last_by_stream),
    }
    return errors, summary
