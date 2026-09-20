"""Build a bounded, typed Parquet sample through the production normalizer."""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.audit_stage1_sources import find_seven_zip  # noqa: E402
from stage1.normalization import NormalizationResult, normalize_chunks  # noqa: E402


DATASET = ROOT / "dataset"
DEFAULT_OUTPUT = ROOT / "output" / "stage1" / "normalized_sample.parquet"
CHANNELS_PER_TYPE = 2
CHUNK_SIZE = 10_000

SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("source_row", pa.int64()),
        ("disposition", pa.string()),
        ("duplicate_of_source", pa.string()),
        ("duplicate_of_source_row", pa.int64()),
        ("event_id", pa.string()),
        ("channel_id", pa.string()),
        ("timestamp", pa.timestamp("us")),
        ("sensor_type", pa.string()),
        ("raw_value", pa.string()),
        ("numeric_value", pa.float64()),
        ("alarm", pa.bool_()),
        ("object_id", pa.string()),
        ("quality_flags", pa.list_(pa.string())),
    ]
)


def read_channel_types(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["ид_канала_данных"]: row["тип_датчика"] for row in csv.DictReader(stream)}


def iter_csv(path: Path, max_rows: int) -> Iterator[dict[str, str]]:
    """Read a plain CSV or a single-member 7z without extracting to disk."""

    process = None
    binary = None
    if path.suffix.casefold() == ".7z":
        process = subprocess.Popen(
            [find_seven_zip(), "x", "-so", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert process.stdout is not None
        binary = process.stdout
    else:
        binary = path.open("rb")
    text_stream = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
    try:
        for index, row in enumerate(csv.DictReader(text_stream), start=1):
            if index > max_rows:
                break
            row["__source__"] = path.name
            row["__source_row__"] = index + 1
            yield row
    finally:
        text_stream.close()
        if process is not None:
            process.terminate()
            process.wait()


def selected_chunks(
    rows: Iterator[dict[str, str]],
    channel_types: dict[str, str],
    selected: dict[str, set[str]],
    channels_per_type: int,
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[list[dict[str, str]]]:
    """Adaptively select at most N observed channels of every known type."""

    chunk: list[dict[str, str]] = []
    for row in rows:
        channel = row["ид_канала_данных"].strip()
        sensor_type = channel_types.get(channel)
        if sensor_type is None:
            continue
        type_channels = selected[sensor_type]
        if channel not in type_channels and len(type_channels) >= channels_per_type:
            continue
        type_channels.add(channel)
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def records_to_batch(records: list[NormalizationResult]) -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist([record.to_record() for record in records], schema=SCHEMA)


def build_sample(
    inputs: list[Path],
    output: Path,
    max_input_rows: int,
    channels_per_type: int,
    *,
    dictionary_path: Path | None = None,
) -> dict:
    channel_types = read_channel_types(
        dictionary_path or DATASET / "справочник_каналов_датчиков.csv"
    )
    all_types = sorted(set(channel_types.values()))
    selected: dict[str, set[str]] = collections.defaultdict(set)
    # The rare type is absent from ordinary prefixes.  Seed its known IDs so a
    # bounded 2025/2026 scan can capture it without selecting unrelated rows.
    for channel, sensor_type in channel_types.items():
        if sensor_type == "9-секционный люк" and len(selected[sensor_type]) < channels_per_type:
            selected[sensor_type].add(channel)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output, SCHEMA, compression="zstd")
    row_counts: collections.Counter[str] = collections.Counter()
    dispositions: collections.Counter[str] = collections.Counter()
    total = 0
    try:

        def all_chunks() -> Iterator[list[dict[str, str]]]:
            for path in inputs:
                yield from selected_chunks(
                    iter_csv(path, max_input_rows),
                    channel_types,
                    selected,
                    channels_per_type,
                )

        pending: list[NormalizationResult] = []
        for result in normalize_chunks(all_chunks(), channel_types, "combined-inputs"):
            pending.append(result)
            dispositions[result.disposition] += 1
            if result.event is not None:
                row_counts[result.event.sensor_type] += 1
            total += 1
            if len(pending) >= CHUNK_SIZE:
                writer.write_batch(records_to_batch(pending))
                pending.clear()
        if pending:
            writer.write_batch(records_to_batch(pending))
    finally:
        writer.close()

    manifest = {
        "inputs": [path.name for path in inputs],
        "max_input_rows_per_source": max_input_rows,
        "channels_per_type_limit": channels_per_type,
        "output": str(output),
        "output_rows": total,
        "dispositions": dict(dispositions),
        "all_dictionary_types": all_types,
        "selected_channels_by_type": {
            sensor_type: sorted(selected[sensor_type]) for sensor_type in all_types
        },
        "rows_by_type": {sensor_type: row_counts[sensor_type] for sensor_type in all_types},
        "types_without_output_rows": [
            sensor_type for sensor_type in all_types if not row_counts[sensor_type]
        ],
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        dest="inputs",
        help="CSV or 7z source; repeat for several sources",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-input-rows", type=int, default=250_000)
    parser.add_argument("--channels-per-type", type=int, default=CHANNELS_PER_TYPE)
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=DATASET / "справочник_каналов_датчиков.csv",
    )
    args = parser.parse_args()
    if args.max_input_rows <= 0 or args.channels_per_type <= 0:
        parser.error("row and channel limits must be positive")
    if not args.inputs:
        args.inputs = [DATASET / "журнал_событий_пример.csv"]
    return args


def main() -> None:
    args = parse_args()
    manifest = build_sample(
        args.inputs,
        args.output,
        args.max_input_rows,
        args.channels_per_type,
        dictionary_path=args.dictionary,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
