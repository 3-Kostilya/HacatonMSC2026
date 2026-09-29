"""Build the R1 technical state-mapping audit from the published M1 clean data.

Only dictionary candidates and alarm consistency are reported. This command
does not create semantic categories, state episodes, or a forecasting target.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import psutil
except ImportError:  # Resource telemetry must not become a required project dependency.
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.state_mapping import (  # noqa: E402
    MAPPING_VERSION,
    build_audit_table,
    load_state_dictionary,
)
from stage1.ingestion.dictionaries import load_dictionaries  # noqa: E402


DEFINITION_SCHEMA = pa.schema(
    [
        pa.field("sensor_type", pa.string(), nullable=False),
        pa.field("state_set_id", pa.string(), nullable=False),
        pa.field("state_text", pa.string(), nullable=False),
        pa.field("expected_alarm", pa.bool_(), nullable=False),
        pa.field("source_rows", pa.list_(pa.int32()), nullable=False),
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_clean_files(m1_dir: Path, quality: dict[str, Any]) -> list[Path]:
    """Use exactly the physical clean files listed by the M1 quality report."""

    expected: dict[str, int] = {}
    for item in quality["output_files"]:
        raw = item["path"].replace("\\", "/")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe M1 output path: {raw}")
        if relative.parts[0] != "clean":
            continue
        if (
            len(relative.parts) != 4
            or not relative.parts[1].startswith("year=")
            or not relative.parts[2].startswith("month=")
            or not relative.parts[3].endswith(".parquet")
        ):
            raise ValueError(f"invalid M1 clean partition path: {raw}")
        if raw in expected or not isinstance(item["bytes"], int) or item["bytes"] <= 0:
            raise ValueError(f"duplicate or invalid M1 output entry: {raw}")
        expected[raw] = item["bytes"]
    if not expected:
        raise ValueError("M1 quality report lists no clean Parquet files")
    actual = {
        path.relative_to(m1_dir).as_posix(): path for path in (m1_dir / "clean").rglob("*.parquet")
    }
    if set(actual) != set(expected):
        raise ValueError("M1 clean Parquet inventory differs from its quality report")
    for name, path in actual.items():
        if path.stat().st_size != expected[name]:
            raise ValueError(f"M1 clean Parquet byte size differs: {name}")
    return [actual[name] for name in sorted(actual)]


def _read_clean_aggregates(files: list[Path], spill_dir: Path) -> list[dict[str, Any]]:
    """Aggregate in DuckDB so raw history never becomes a Python list."""

    spill_dir.mkdir(parents=True, exist_ok=True)
    database = duckdb.connect(":memory:")
    try:
        database.execute("SET memory_limit='4GB'")
        database.execute("SET threads=2")
        database.execute("SET temp_directory=?", [str(spill_dir)])
        result = database.execute(
            """
            SELECT CAST(year(timestamp) AS SMALLINT) AS year,
                   sensor_type,
                   value_state AS state_text_raw,
                   alarm AS observed_alarm,
                   count(*)::BIGINT AS row_count
            FROM read_parquet(?, hive_partitioning=false)
            GROUP BY 1, 2, 3, 4
            ORDER BY 1, 2, 3, 4
            """,
            [[str(path) for path in files]],
        ).fetchall()
    finally:
        database.close()
    return [
        {
            "year": year,
            "sensor_type": sensor_type,
            "state_text_raw": state_text_raw,
            "observed_alarm": observed_alarm,
            "row_count": row_count,
        }
        for year, sensor_type, state_text_raw, observed_alarm, row_count in result
    ]


def _weighted_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row[key])] += int(row["row_count"])
    return dict(sorted(counts.items()))


def _report(
    table: pa.Table,
    numeric_rows: list[dict[str, Any]],
    *,
    accepted_rows: int,
    dictionary: Any,
    input_manifest_sha256: str,
    quality_sha256: str,
    clean_file_count: int,
    parquet_sha256: str,
    elapsed_seconds: float,
    peak_working_set_bytes: int | None,
) -> dict[str, Any]:
    mapped = table.to_pylist()
    text_total = sum(row["row_count"] for row in mapped)
    numeric_total = sum(int(row["row_count"]) for row in numeric_rows)
    if text_total + numeric_total != accepted_rows:
        raise AssertionError("text and numeric R1 totals do not conserve M1 accepted rows")
    by_year: dict[str, Counter[str]] = defaultdict(Counter)
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    status_counts: Counter[str] = Counter()
    consistency_counts: Counter[str] = Counter()
    unmapped_pairs: Counter[tuple[str | None, str, str]] = Counter()
    neispraven_status_counts: Counter[str] = Counter()
    for row in mapped:
        count = row["row_count"]
        status = row["match_status"]
        status_counts[status] += count
        consistency_counts[row["alarm_consistency"]] += count
        by_year[str(row["year"])][status] += count
        by_type[row["sensor_type"] or "<unknown>"][status] += count
        if status in {"unmapped_state", "unmapped_type"}:
            unmapped_pairs[(row["sensor_type"], row["state_text_raw"], status)] += count
        if row["state_text_raw"] == "Неисправен":
            neispraven_status_counts[status] += count
    conflicts: dict[tuple[str, str, str], set[bool]] = defaultdict(set)
    ambiguous_alarm_keys: dict[tuple[str, str], set[bool]] = defaultdict(set)
    for definition in dictionary.definitions:
        conflicts[(definition.sensor_type, definition.state_set_id, definition.state_text)].add(
            definition.expected_alarm
        )
        ambiguous_alarm_keys[(definition.sensor_type, definition.state_text)].add(
            definition.expected_alarm
        )
    return {
        "schema_version": MAPPING_VERSION,
        "status": "complete",
        "aggregate_source": "direct_m1_clean_parquet",
        "data_quality_sha256": quality_sha256,
        "clean_file_count": clean_file_count,
        "input_manifest_sha256": input_manifest_sha256,
        "dictionary_sha256": dictionary.sha256,
        "dictionary_original_rows": dictionary.source_row_count,
        "dictionary_unique_full_rows": dictionary.unique_definition_count,
        "dictionary_exact_duplicate_rows": dictionary.duplicate_row_count,
        "dictionary_types": len(dictionary.types),
        "dictionary_state_sets": len({item.state_set_id for item in dictionary.definitions}),
        "conflicting_definition_keys": [
            {"sensor_type": key[0], "state_set_id": key[1], "state_text": key[2]}
            for key, alarms in sorted(conflicts.items())
            if len(alarms) > 1
        ],
        "type_state_keys_with_multiple_expected_alarms": [
            {"sensor_type": key[0], "state_text": key[1]}
            for key, alarms in sorted(ambiguous_alarm_keys.items())
            if len(alarms) > 1
        ],
        "accepted_rows": accepted_rows,
        "text_rows": text_total,
        "numeric_rows_not_applicable": numeric_total,
        "text_aggregate_rows": len(mapped),
        "numeric_aggregate_rows": len(numeric_rows),
        "match_status_rows": dict(sorted(status_counts.items())),
        "alarm_consistency_rows": dict(sorted(consistency_counts.items())),
        "exact_text_neispraven_by_match_status": dict(sorted(neispraven_status_counts.items())),
        "top_unmapped_type_state": [
            {
                "sensor_type": sensor_type,
                "state_text_raw": state_text,
                "match_status": status,
                "rows": count,
            }
            for (sensor_type, state_text, status), count in sorted(
                unmapped_pairs.items(),
                key=lambda item: (-item[1], item[0][0] or "", item[0][1], item[0][2]),
            )[:20]
        ],
        "by_year_status_rows": {
            year: dict(sorted(counts.items())) for year, counts in sorted(by_year.items())
        },
        "by_type_status_rows": {
            name: dict(sorted(counts.items())) for name, counts in sorted(by_type.items())
        },
        "numeric_by_year_rows": _weighted_counts(numeric_rows, "year"),
        "state_mapping_audit_sha256": parquet_sha256,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "peak_process_working_set_bytes": peak_working_set_bytes,
        "limitations": [
            "candidate sets do not establish a historical channel-to-state-set link",
            "observed alarm is checked only after candidate lookup, never used to choose a candidate",
            "dictionary candidates and alarm consistency are not semantic failure labels",
            "numeric values have no text-state mapping; their counts are reported separately",
            "2021 is excluded from the supplied M1 history",
        ],
    }


def build(
    *,
    input_manifest: Path,
    state_dictionary: Path,
    output: Path,
    channels_dictionary: Path | None = None,
    objects_dictionary: Path | None = None,
) -> dict[str, Any]:
    """Publish one immutable R1 audit directory, or leave an in-progress directory."""

    started = time.monotonic()
    input_manifest = input_manifest.resolve()
    state_dictionary = state_dictionary.resolve()
    output = output.resolve()
    pending = output.with_name(output.name + ".inprogress")
    if output.exists() or pending.exists():
        raise FileExistsError("R1 output or .inprogress directory exists; use a new path")
    m1 = json.loads(input_manifest.read_text(encoding="utf-8"))
    if m1.get("schema_version") != "ingestion-v1" or m1.get("status") != "complete":
        raise ValueError("R1 requires a complete ingestion-v1 manifest")
    if m1.get("scope") != "full_supplied_sources":
        raise ValueError("R1 requires the full supplied M1 sources")
    m1_dir = input_manifest.parent
    quality_path = m1_dir / "data_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality.get("scope") != m1["scope"] or quality.get("input_rows") != m1.get("input_rows"):
        raise ValueError("M1 manifest and quality report disagree")
    accepted_rows = int(quality["dispositions"]["accepted"])
    clean_files = _validated_clean_files(m1_dir, quality)
    quality_sha256 = _sha256(quality_path)
    input_manifest_sha256 = _sha256(input_manifest)
    m1_dictionaries = m1.get("dictionaries", [])
    if len(m1_dictionaries) != 2:
        raise ValueError("R1 requires both M1 channel and object dictionaries")
    verified_m1_dictionaries = []
    for role, item, override in zip(
        ("channels", "objects"),
        m1_dictionaries,
        (channels_dictionary, objects_dictionary),
    ):
        path = (override or Path(item["path"])).resolve()
        actual_sha = _sha256(path)
        if actual_sha != item["sha256"]:
            raise ValueError(f"M1 dictionary changed since ingestion: {path}")
        verified_m1_dictionaries.append({"role": role, "path": str(path), "sha256": actual_sha})
    dictionary = load_state_dictionary(state_dictionary)
    channel_rows, object_rows, source_dictionary_audit = load_dictionaries(
        Path(verified_m1_dictionaries[0]["path"]),
        Path(verified_m1_dictionaries[1]["path"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pending.mkdir()
    aggregates = _read_clean_aggregates(clean_files, pending / "spill")
    if sum(int(row["row_count"]) for row in aggregates) != accepted_rows:
        raise AssertionError("R1 aggregates do not conserve M1 accepted rows")
    if any(int(row["year"]) == 2021 for row in aggregates):
        raise ValueError("R1 aggregates unexpectedly contain excluded year 2021")
    numeric = [row for row in aggregates if row["state_text_raw"] is None]
    text = [row for row in aggregates if row["state_text_raw"] is not None]
    table = build_audit_table(text, dictionary, input_manifest_sha256=input_manifest_sha256)
    audit_path = pending / "state_mapping_audit.parquet"
    pq.write_table(table, audit_path, compression="zstd")
    definitions = pa.Table.from_pylist(
        [
            {
                "sensor_type": item.sensor_type,
                "state_set_id": item.state_set_id,
                "state_text": item.state_text,
                "expected_alarm": item.expected_alarm,
                "source_rows": list(item.source_rows),
            }
            for item in dictionary.definitions
        ],
        schema=DEFINITION_SCHEMA,
    )
    definitions_path = pending / "dictionary_definitions.parquet"
    pq.write_table(definitions, definitions_path, compression="zstd")
    if psutil is not None:
        memory = psutil.Process().memory_info()
        peak_working_set_bytes = getattr(memory, "peak_wset", memory.rss)
    else:
        peak_working_set_bytes = None
    report = _report(
        table,
        numeric,
        accepted_rows=accepted_rows,
        dictionary=dictionary,
        input_manifest_sha256=input_manifest_sha256,
        quality_sha256=quality_sha256,
        clean_file_count=len(clean_files),
        parquet_sha256=_sha256(audit_path),
        elapsed_seconds=time.monotonic() - started,
        peak_working_set_bytes=peak_working_set_bytes,
    )
    channel_types = {row["sensor_type"] for row in channel_rows if row["sensor_type"]}
    report["channel_dictionary_rows"] = len(channel_rows)
    report["object_dictionary_rows"] = len(object_rows)
    report["channel_dictionary_types"] = len(channel_types)
    report["channel_types_absent_from_state_dictionary"] = sorted(channel_types - dictionary.types)
    report["state_types_absent_from_channel_dictionary"] = sorted(dictionary.types - channel_types)
    report["object_mapping_available"] = source_dictionary_audit["object_mapping_available"]
    report_path = pending / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": MAPPING_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": input_manifest_sha256,
        "state_dictionary": str(state_dictionary),
        "state_dictionary_sha256": dictionary.sha256,
        "verified_m1_dictionaries": verified_m1_dictionaries,
        "aggregate_source": "direct_m1_clean_parquet",
        "data_quality_sha256": quality_sha256,
        "clean_file_count": len(clean_files),
        "files": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (audit_path, definitions_path, report_path)
        },
        "row_count": table.num_rows,
        "elapsed_seconds": report["elapsed_seconds"],
        "peak_process_working_set_bytes": report["peak_process_working_set_bytes"],
    }
    (pending / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--state-dictionary", type=Path, required=True)
    parser.add_argument("--channels-dictionary", type=Path)
    parser.add_argument("--objects-dictionary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(
        input_manifest=args.input_manifest,
        state_dictionary=args.state_dictionary,
        output=args.output,
        channels_dictionary=args.channels_dictionary,
        objects_dictionary=args.objects_dictionary,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": manifest["status"],
                "row_count": manifest["row_count"],
                "elapsed_seconds": manifest["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
