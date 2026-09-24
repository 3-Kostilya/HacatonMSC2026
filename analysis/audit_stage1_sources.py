"""Fast, bounded audit of Stage 1 data sources.

The default probe reads only a prefix of every event source.  Use ``--full``
explicitly when an exact full-history scan is intended.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import BinaryIO, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "output" / "stage1" / "source_audit.json"
EVENT_SCHEMA = [
    "ид_события",
    "ид_канала_данных",
    "дата",
    "время",
    "тревожное",
    "значение_датчика",
]

GROUP_TYPES = {
    "Числовые измерения среды": {"Датчик температуры", "Газовый датчик"},
    "Пожарные извещатели": {"Датчик дыма", "Тепловой датчик"},
    "Охранные и контактные каналы": {
        "Датчик движения",
        "КД АВ",
        "КД Дверь",
        "КД Люк",
        "9-секционный люк",
        "Стекло",
    },
    "Контроль затопления": {"Датчик затопления"},
    "Состояние оборудования": {"Состояние насоса", "Состояние вентилятора"},
    "Питание и состояние аппаратуры": {"ИБП", "Состояние фазы", "Состояние УИР-Р"},
    "Управление, ручные сигналы и режим охраны": {
        "Переключатель",
        "Ручной извещатель",
        "Состояние охраны",
    },
}
TYPE_TO_GROUP = {
    sensor_type: group for group, types in GROUP_TYPES.items() for sensor_type in types
}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def find_seven_zip() -> str:
    configured = os.environ.get("SEVEN_ZIP")
    candidates = [configured] if configured else ["7zz", "7z", r"C:/Program Files/7-Zip/7z.exe"]
    for candidate in candidates:
        if candidate and (resolved := shutil.which(candidate)):
            return resolved
    raise FileNotFoundError("7-Zip not found; install 7z/7zz or set SEVEN_ZIP")


def parse_7z_listing(text: str) -> list[dict[str, str]]:
    """Return file-member records from a ``7z l -slt`` listing."""
    tail = text.split("----------", 1)
    if len(tail) != 2:
        raise ValueError("Unexpected 7-Zip listing: member delimiter absent")
    records = []
    for block in re.split(r"\r?\n\r?\n", tail[1].strip()):
        record = {}
        for line in block.splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
                record[key] = value
        if record.get("Path"):
            records.append(record)
    return records


def detect_utf8_encoding(prefix: bytes) -> str:
    if prefix.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    prefix.decode("utf-8", errors="strict")
    return "utf-8"


def read_dictionary(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    prefix = path.read_bytes()[:65536]
    encoding = detect_utf8_encoding(prefix)
    with path.open(encoding=encoding, newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return list(reader.fieldnames or []), rows, encoding


def validate_grouping(sensor_types: Iterable[str]) -> dict[str, list[str]]:
    actual = set(sensor_types)
    configured = set(TYPE_TO_GROUP)
    return {
        "unmapped_dictionary_types": sorted(actual - configured),
        "configured_types_absent_from_dictionary": sorted(configured - actual),
    }


def audit_object_links(rows: list[dict[str, str]]) -> dict:
    by_id = {row["ид_объект"]: row for row in rows}
    missing_parent_ids = sorted(
        {row["родитель"] for row in rows if row["родитель"] and row["родитель"] not in by_id}
    )
    cycle_start_ids = []
    for start in by_id:
        seen = set()
        current = start
        while current in by_id:
            if current in seen:
                cycle_start_ids.append(start)
                break
            seen.add(current)
            current = by_id[current]["родитель"]
    return {"missing_parent_ids": missing_parent_ids, "cycle_start_ids": cycle_start_ids}


def _probe_csv_stream(
    binary: BinaryIO,
    type_by_channel: dict[str, str],
    max_rows: int | None,
) -> dict:
    started = time.perf_counter()
    text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
    reader = csv.DictReader(text)
    schema = list(reader.fieldnames or [])
    if schema != EVENT_SCHEMA:
        raise ValueError(f"Unexpected event schema: {schema!r}")
    types: collections.Counter[str] = collections.Counter()
    groups: collections.Counter[str] = collections.Counter()
    alarms: collections.Counter[str] = collections.Counter()
    values_numeric = 0
    unknown_channels = 0
    rows = 0
    min_date = None
    max_date = None
    for row in reader:
        rows += 1
        channel = row["ид_канала_данных"]
        sensor_type = type_by_channel.get(channel)
        if sensor_type is None:
            unknown_channels += 1
        else:
            types[sensor_type] += 1
            groups[TYPE_TO_GROUP.get(sensor_type, "Не сопоставлено")] += 1
        date = row["дата"]
        min_date = date if min_date is None or date < min_date else min_date
        max_date = date if max_date is None or date > max_date else max_date
        alarms[row["тревожное"]] += 1
        try:
            float(row["значение_датчика"].replace(",", "."))
            values_numeric += 1
        except ValueError:
            pass
        if max_rows is not None and rows >= max_rows:
            break
    elapsed = time.perf_counter() - started
    # Detach so closing the subprocess pipe remains the caller's responsibility.
    text.detach()
    return {
        "schema": schema,
        "encoding": "utf-8-sig (UTF-8 accepted with optional BOM)",
        "rows_probed": rows,
        "probe_limit": max_rows,
        "complete_scan": max_rows is None,
        "elapsed_seconds": round(elapsed, 6),
        "rows_per_second": round(rows / elapsed, 1) if elapsed else None,
        "sample_date_min": min_date,
        "sample_date_max": max_date,
        "numeric_value_rows": values_numeric,
        "unknown_channel_rows": unknown_channels,
        "alarm_values": dict(alarms),
        "rows_by_type": dict(types),
        "rows_by_group": dict(groups),
    }


def probe_plain_csv(path: Path, type_by_channel: dict[str, str], max_rows: int | None) -> dict:
    with path.open("rb") as stream:
        return _probe_csv_stream(stream, type_by_channel, max_rows)


def probe_archive(
    path: Path, seven_zip: str, type_by_channel: dict[str, str], max_rows: int | None
) -> dict:
    listing_text = subprocess.check_output(
        [seven_zip, "l", "-slt", str(path)], text=True, encoding="utf-8"
    )
    members = parse_7z_listing(listing_text)
    csv_members = [m for m in members if m["Path"].lower().endswith(".csv")]
    if len(csv_members) != 1:
        raise ValueError(f"{path.name}: expected one CSV member, found {len(csv_members)}")
    process = subprocess.Popen(
        [seven_zip, "x", "-so", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    try:
        result = _probe_csv_stream(process.stdout, type_by_channel, max_rows)
    finally:
        process.stdout.close()
        if max_rows is not None:
            process.terminate()
        exit_code = process.wait()
    # A deliberately bounded probe terminates 7-Zip early; a full scan must succeed.
    if max_rows is None and exit_code != 0:
        raise RuntimeError(f"7-Zip exited with {exit_code} while reading {path.name}")
    result["archive_member"] = csv_members[0]
    result["archive_exit_code"] = exit_code
    result["bounded_probe_terminated_extractor"] = max_rows is not None
    return result


def audit(max_rows: int | None, output: Path, hash_archives: bool = True) -> dict:
    channels_path = DATASET / "справочник_каналов_датчиков.csv"
    objects_path = DATASET / "справочник_объектов_диспетчер.csv"
    channel_schema, channels, channel_encoding = read_dictionary(channels_path)
    object_schema, objects, object_encoding = read_dictionary(objects_path)
    channel_ids = [r["ид_канала_данных"] for r in channels]
    object_ids = [r["ид_объект"] for r in objects]
    type_by_channel = {r["ид_канала_данных"]: r["тип_датчика"] for r in channels}
    type_counts = collections.Counter(r["тип_датчика"] for r in channels)
    seven_zip = find_seven_zip()
    seven_version_text = subprocess.check_output(
        [seven_zip, "i"], text=True, encoding="utf-8", errors="replace"
    )
    seven_version = next(
        (line.strip() for line in seven_version_text.splitlines() if line.startswith("7-Zip")),
        "unknown",
    )
    sources = []
    paths = [DATASET / "журнал_событий_пример.csv", *sorted(DATASET.glob("ext-journal-*.7z"))]
    for path in paths:
        stat = path.stat()
        base = {
            "file": path.name,
            "bytes": stat.st_size,
            "sha256": sha256_file(path) if (hash_archives or path.suffix != ".7z") else None,
        }
        if path.suffix == ".7z":
            base.update(probe_archive(path, seven_zip, type_by_channel, max_rows))
        else:
            base.update(probe_plain_csv(path, type_by_channel, max_rows))
        sources.append(base)
        print(
            f"{path.name}: {base['rows_probed']:,} rows, "
            f"{base['rows_per_second']:,.0f} rows/s, {len(base['rows_by_group'])}/7 groups",
            flush=True,
        )
    observed_types = sorted({key for source in sources for key in source["rows_by_type"]})
    observed_groups = sorted({key for source in sources for key in source["rows_by_group"]})
    archive_uncompressed_bytes = sum(
        int(source["archive_member"]["Size"]) for source in sources if "archive_member" in source
    )
    total_probe_seconds = sum(source["elapsed_seconds"] for source in sources)
    total_probe_rows = sum(source["rows_probed"] for source in sources)
    years = [int(match.group(1)) for p in paths if (match := re.search(r"(\d{4})", p.name))]
    result = {
        "mode": "full" if max_rows is None else "bounded_probe",
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seven_zip": {"path": seven_zip, "version": seven_version},
        "availability": {
            "archive_years": years,
            "missing_years_inside_range": sorted(
                set(range(min(years), max(years) + 1)) - set(years)
            ),
        },
        "channel_dictionary": {
            "file": channels_path.name,
            "bytes": channels_path.stat().st_size,
            "sha256": sha256_file(channels_path),
            "encoding": channel_encoding,
            "schema": channel_schema,
            "rows": len(channels),
            "unique_channel_ids": len(set(channel_ids)),
            "duplicate_channel_ids": len(channel_ids) - len(set(channel_ids)),
            "object_id_columns": [c for c in channel_schema if "объект" in c.lower()],
            "sensor_type_counts": dict(type_counts),
            "grouping_check": validate_grouping(type_counts),
        },
        "object_dictionary": {
            "file": objects_path.name,
            "bytes": objects_path.stat().st_size,
            "sha256": sha256_file(objects_path),
            "encoding": object_encoding,
            "schema": object_schema,
            "rows": len(objects),
            "unique_object_ids": len(set(object_ids)),
            "duplicate_object_ids": len(object_ids) - len(set(object_ids)),
            "link_check": audit_object_links(objects),
        },
        "probe_summary": {
            "rows": total_probe_rows,
            "elapsed_seconds": round(total_probe_seconds, 6),
            "weighted_rows_per_second": round(total_probe_rows / total_probe_seconds, 1),
            "archive_compressed_bytes": sum(
                source["bytes"] for source in sources if source["file"].endswith(".7z")
            ),
            "archive_uncompressed_bytes": archive_uncompressed_bytes,
            "observed_types": observed_types,
            "dictionary_types_not_observed_in_bounded_probe": sorted(
                set(type_counts) - set(observed_types)
            ),
            "observed_groups": observed_groups,
            "configured_groups_not_observed": sorted(set(GROUP_TYPES) - set(observed_groups)),
            "unknown_channel_rows": sum(source["unknown_channel_rows"] for source in sources),
        },
        "event_sources": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=100_000, help="rows per event source")
    parser.add_argument("--full", action="store_true", help="scan every row (potentially hours)")
    parser.add_argument(
        "--skip-archive-hashes", action="store_true", help="faster, weaker manifest"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    return args


def main() -> None:
    args = parse_args()
    audit(None if args.full else args.max_rows, args.output, not args.skip_archive_hashes)


if __name__ == "__main__":
    main()
