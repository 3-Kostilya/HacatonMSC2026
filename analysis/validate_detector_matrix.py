"""Validate the detector applicability matrix against the live channel dictionary."""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "stage1/config/type_detector_matrix.yaml"
REQUIRED_TYPE_COUNT = 19
REQUIRED_MODES = {"numeric", "discrete", "context"}
REQUIRED_TYPE_FIELDS = {
    "sensor_type",
    "family",
    "expected_channels",
    "modes",
    "features",
    "mandatory_checks",
    "unknown_if",
    "inapplicability_rule",
    "interpretation_guard",
}


class MatrixValidationError(ValueError):
    """Raised when the matrix is incomplete or inconsistent with the dictionary."""


def load_matrix(path: Path) -> dict:
    """Load the JSON-compatible YAML without adding a third-party dependency."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MatrixValidationError(f"Matrix is not valid JSON-compatible YAML: {exc}") from exc


def dictionary_counts(path: Path, type_column: str) -> collections.Counter[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or type_column not in reader.fieldnames:
            raise MatrixValidationError(
                f"Dictionary {path} does not contain required column {type_column!r}"
            )
        counts = collections.Counter(row[type_column].strip() for row in reader)
    if "" in counts:
        raise MatrixValidationError("Dictionary contains a blank sensor type")
    return counts


def validate_matrix(matrix_path: Path = DEFAULT_MATRIX, root: Path = ROOT) -> dict:
    matrix = load_matrix(matrix_path)
    errors: list[str] = []

    modes = matrix.get("processing_modes")
    if not isinstance(modes, list) or set(modes) != REQUIRED_MODES or len(modes) != 3:
        errors.append(f"processing_modes must contain exactly {sorted(REQUIRED_MODES)}")

    statuses = matrix.get("mode_statuses")
    if not isinstance(statuses, list) or not statuses:
        errors.append("mode_statuses must be a non-empty list")
        statuses = []
    status_set = set(statuses)

    feature_catalog = matrix.get("feature_catalog")
    check_catalog = matrix.get("check_catalog")
    if not isinstance(feature_catalog, dict) or not feature_catalog:
        errors.append("feature_catalog must be a non-empty object")
        feature_catalog = {}
    if not isinstance(check_catalog, dict) or not check_catalog:
        errors.append("check_catalog must be a non-empty object")
        check_catalog = {}

    entries = matrix.get("types")
    if not isinstance(entries, list):
        errors.append("types must be a list")
        entries = []
    if len(entries) != REQUIRED_TYPE_COUNT:
        errors.append(
            f"matrix must contain exactly {REQUIRED_TYPE_COUNT} type rows, got {len(entries)}"
        )

    names = [entry.get("sensor_type") for entry in entries if isinstance(entry, dict)]
    duplicates = sorted(name for name, count in collections.Counter(names).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate sensor types: {duplicates}")

    source = matrix.get("source_dictionary")
    type_column = matrix.get("dictionary_type_column")
    if not isinstance(source, str) or not source:
        errors.append("source_dictionary must be a non-empty string")
        dictionary = collections.Counter()
    elif not isinstance(type_column, str) or not type_column:
        errors.append("dictionary_type_column must be a non-empty string")
        dictionary = collections.Counter()
    else:
        dictionary_path = root / source
        if not dictionary_path.is_file():
            errors.append(f"source dictionary does not exist: {dictionary_path}")
            dictionary = collections.Counter()
        else:
            dictionary = dictionary_counts(dictionary_path, type_column)

    if len(dictionary) != REQUIRED_TYPE_COUNT:
        errors.append(
            f"dictionary must contain exactly {REQUIRED_TYPE_COUNT} types, got {len(dictionary)}"
        )
    matrix_names = set(names)
    dictionary_names = set(dictionary)
    if missing := sorted(dictionary_names - matrix_names):
        errors.append(f"dictionary types missing from matrix: {missing}")
    if extra := sorted(matrix_names - dictionary_names):
        errors.append(f"matrix types absent from dictionary: {extra}")

    for index, entry in enumerate(entries):
        label = (
            entry.get("sensor_type", f"row {index}") if isinstance(entry, dict) else f"row {index}"
        )
        if not isinstance(entry, dict):
            errors.append(f"{label}: type row must be an object")
            continue
        missing_fields = sorted(REQUIRED_TYPE_FIELDS - set(entry))
        if missing_fields:
            errors.append(f"{label}: missing fields {missing_fields}")
        entry_modes = entry.get("modes")
        if not isinstance(entry_modes, dict) or set(entry_modes) != REQUIRED_MODES:
            errors.append(f"{label}: modes must define exactly {sorted(REQUIRED_MODES)}")
        else:
            invalid_statuses = sorted(set(entry_modes.values()) - status_set)
            if invalid_statuses:
                errors.append(f"{label}: invalid mode statuses {invalid_statuses}")
            if all(value == "not_applicable" for value in entry_modes.values()):
                errors.append(f"{label}: at least one processing mode must be usable")

        features = entry.get("features")
        if not isinstance(features, list) or not features:
            errors.append(f"{label}: features must be a non-empty list")
        else:
            unknown_features = sorted(set(features) - set(feature_catalog))
            if unknown_features:
                errors.append(f"{label}: unknown features {unknown_features}")

        checks = entry.get("mandatory_checks")
        if not isinstance(checks, list) or not checks:
            errors.append(f"{label}: mandatory_checks must be a non-empty list")
        else:
            unknown_checks = sorted(set(checks) - set(check_catalog))
            if unknown_checks:
                errors.append(f"{label}: unknown mandatory checks {unknown_checks}")
            for universal in (
                "value_format_audit",
                "history_sufficiency",
                "cause_remains_hypothesis",
            ):
                if universal not in checks:
                    errors.append(f"{label}: missing universal mandatory check {universal}")

        if not isinstance(entry.get("unknown_if"), list) or not entry.get("unknown_if"):
            errors.append(f"{label}: unknown_if must be a non-empty list")
        for field in ("inapplicability_rule", "interpretation_guard", "family"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                errors.append(f"{label}: {field} must be a non-empty string")

        expected = entry.get("expected_channels")
        actual = dictionary.get(label)
        if not isinstance(expected, int) or expected < 0:
            errors.append(f"{label}: expected_channels must be a non-negative integer")
        elif actual is not None and expected != actual:
            errors.append(f"{label}: expected_channels={expected}, dictionary has {actual}")

    if errors:
        raise MatrixValidationError("Detector matrix validation failed:\n- " + "\n- ".join(errors))

    return {
        "types": len(entries),
        "channels": sum(dictionary.values()),
        "modes": sorted(REQUIRED_MODES),
        "features": len(feature_catalog),
        "checks": len(check_catalog),
        "matrix": str(matrix_path),
        "dictionary": str(root / source),
    }


def main() -> int:
    path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_MATRIX
    try:
        result = validate_matrix(path)
    except (MatrixValidationError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
