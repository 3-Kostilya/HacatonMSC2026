"""Load the small Stage 1 reference dictionaries into canonical records."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any


CHANNEL_REQUIRED_COLUMNS = (
    "ид_канала_данных",
    "тип_инж_системы",
    "тип_датчика",
    "тег_инженерной_системы",
    "название_датчика",
)
CHANNEL_OBJECT_ID_COLUMN = "ид_объект"
OBJECT_REQUIRED_COLUMNS = (
    "ид_объект",
    "иерархия_уровень",
    "родитель",
    "вид_объекта",
    "диспетчерское_название_объекта",
)


def _read_csv(
    path: Path, required_columns: tuple[str, ...]
) -> tuple[list[str], list[dict[str, str]]]:
    """Read a UTF-8 CSV and reject an incomplete or ambiguous header."""

    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Dictionary {path} has no header")
        duplicates = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
        if duplicates:
            raise ValueError(f"Dictionary {path} has duplicate columns: {duplicates}")
        missing = [name for name in required_columns if name not in fieldnames]
        if missing:
            raise ValueError(f"Dictionary {path} is missing required columns: {missing}")
        rows = []
        for record_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Dictionary {path} record {record_number} has too many fields")
            if any(row[column] is None for column in fieldnames):
                raise ValueError(f"Dictionary {path} record {record_number} has too few fields")
            rows.append(row)
        return fieldnames, rows


def _value(row: dict[str, str], column: str) -> str:
    value = row.get(column, "")
    return "" if value is None else value


def _optional_text(row: dict[str, str], column: str) -> str | None:
    value = _value(row, column)
    return value if value.strip() else None


def _required_id(row: dict[str, str], column: str, path: Path, row_number: int) -> str:
    value = _value(row, column).strip()
    if not value:
        raise ValueError(f"Dictionary {path} row {row_number} has an empty {column!r}")
    return value


def _require_unique(rows: list[dict[str, Any]], key: str, path: Path) -> None:
    seen: set[str] = set()
    for row in rows:
        value = row[key]
        if value in seen:
            raise ValueError(f"Dictionary {path} has duplicate {key}: {value!r}")
        seen.add(value)


def _hierarchy_cycles(objects: list[dict[str, str | None]]) -> list[list[str]]:
    """Return each directed parent cycle once, including self-references."""

    parent_by_id = {row["object_id"]: row["parent_object_id"] for row in objects}
    state: dict[str, int] = {}
    cycles: set[tuple[str, ...]] = set()

    for start in parent_by_id:
        if state.get(start, 0) == 2:
            continue
        path: list[str] = []
        positions: dict[str, int] = {}
        current: str | None = start
        while current is not None and current in parent_by_id and state.get(current, 0) != 2:
            if current in positions:
                cycle = path[positions[current] :]
                rotations = [tuple(cycle[index:] + cycle[:index]) for index in range(len(cycle))]
                cycles.add(min(rotations))
                break
            positions[current] = len(path)
            path.append(current)
            state[current] = 1
            current = parent_by_id[current]
        for object_id in path:
            state[object_id] = 2

    return [list(cycle) for cycle in sorted(cycles)]


def load_dictionaries(
    channels_path: Path, objects_path: Path
) -> tuple[list[dict[str, str | None]], list[dict[str, str | None]], dict[str, Any]]:
    """Load channel and object dictionaries without inventing object links.

    Empty optional values become ``None``.  The channel dictionary's object-id
    column is optional because the supplied source dictionary may not contain it.
    """

    channel_columns, raw_channels = _read_csv(channels_path, CHANNEL_REQUIRED_COLUMNS)
    _, raw_objects = _read_csv(objects_path, OBJECT_REQUIRED_COLUMNS)
    object_mapping_available = CHANNEL_OBJECT_ID_COLUMN in channel_columns

    channel_rows: list[dict[str, str | None]] = []
    for row_number, row in enumerate(raw_channels, start=2):
        channel_rows.append(
            {
                "channel_id": _required_id(row, "ид_канала_данных", channels_path, row_number),
                "sensor_type": _value(row, "тип_датчика"),
                "engineering_system_type": _value(row, "тип_инж_системы"),
                "engineering_system_tag": _value(row, "тег_инженерной_системы"),
                "sensor_name": _value(row, "название_датчика"),
                "object_id": (
                    _optional_text(row, CHANNEL_OBJECT_ID_COLUMN).strip()
                    if object_mapping_available
                    and _optional_text(row, CHANNEL_OBJECT_ID_COLUMN) is not None
                    else None
                ),
            }
        )

    object_rows: list[dict[str, str | None]] = []
    for row_number, row in enumerate(raw_objects, start=2):
        object_rows.append(
            {
                "object_id": _required_id(row, "ид_объект", objects_path, row_number),
                "hierarchy_level": _optional_text(row, "иерархия_уровень"),
                "parent_object_id": (
                    _optional_text(row, "родитель").strip()
                    if _optional_text(row, "родитель") is not None
                    else None
                ),
                "object_kind": _optional_text(row, "вид_объекта"),
                "object_name": _optional_text(row, "диспетчерское_название_объекта"),
            }
        )

    _require_unique(channel_rows, "channel_id", channels_path)
    _require_unique(object_rows, "object_id", objects_path)

    object_ids = {row["object_id"] for row in object_rows}
    missing_parent_ids = sorted(
        {
            row["parent_object_id"]
            for row in object_rows
            if row["parent_object_id"] is not None and row["parent_object_id"] not in object_ids
        }
    )
    absent_channel_object_foreign_keys = sum(
        row["object_id"] is not None and row["object_id"] not in object_ids for row in channel_rows
    )
    channel_counts_by_type = dict(
        sorted(Counter(row["sensor_type"] for row in channel_rows).items())
    )
    audit: dict[str, Any] = {
        "total_channels": len(channel_rows),
        "total_objects": len(object_rows),
        "channel_counts_by_type": channel_counts_by_type,
        "object_mapping_available": object_mapping_available,
        "absent_channel_object_foreign_keys": absent_channel_object_foreign_keys,
        "absent_object_parent_foreign_keys": len(missing_parent_ids),
        "missing_parent_ids": missing_parent_ids,
        "hierarchy_cycles": _hierarchy_cycles(object_rows),
    }
    return channel_rows, object_rows, audit
