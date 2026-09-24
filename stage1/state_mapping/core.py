"""Traceable technical join of journal state text to dictionary candidates.

This module never assigns a channel to a state set, interprets an alarm as a
failure, or infers a semantic target. Those decisions belong to R1/B1.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pyarrow as pa


MAPPING_VERSION = "state-mapping-r1-v1"
NORMALIZATION_RULE = "observed_state_strip_only_v1"
MATCH_STATUSES = frozenset(
    {
        "exact_candidate",
        "multiple_candidates",
        "conflicting_definition",
        "unmapped_state",
        "unmapped_type",
    }
)
REQUIRED_COLUMNS = (
    "тип_датчика",
    "ид_набор_состояний",
    "название_состояния",
    "тревожное",
)


STATE_MAPPING_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("config_sha256", pa.string(), nullable=False),
        pa.field("input_manifest_sha256", pa.string(), nullable=False),
        pa.field("dictionary_sha256", pa.string(), nullable=False),
        pa.field("year", pa.int16(), nullable=False),
        pa.field("sensor_type", pa.string()),
        pa.field("state_text_raw", pa.string(), nullable=False),
        pa.field("state_match_key", pa.string(), nullable=False),
        pa.field("observed_alarm", pa.bool_(), nullable=False),
        pa.field("row_count", pa.int64(), nullable=False),
        pa.field("match_status", pa.string(), nullable=False),
        pa.field("candidate_set_ids", pa.list_(pa.string()), nullable=False),
        pa.field("expected_alarm_values", pa.list_(pa.bool_()), nullable=False),
        pa.field("expected_alarm", pa.bool_()),
        pa.field("alarm_consistency", pa.string(), nullable=False),
        pa.field("definition_source_rows", pa.list_(pa.int32()), nullable=False),
        pa.field("normalization_rule", pa.string(), nullable=False),
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdecimal() else (1, value)


@dataclass(frozen=True, slots=True)
class StateDefinition:
    sensor_type: str
    state_set_id: str
    state_text: str
    expected_alarm: bool
    source_rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StateDictionary:
    path: Path
    sha256: str
    source_row_count: int
    definitions: tuple[StateDefinition, ...]
    types: frozenset[str]
    by_key: Mapping[tuple[str, str], tuple[StateDefinition, ...]]

    @property
    def unique_definition_count(self) -> int:
        return len(self.definitions)

    @property
    def duplicate_row_count(self) -> int:
        return self.source_row_count - len(self.definitions)


def load_state_dictionary(path: Path) -> StateDictionary:
    """Load four original CSV fields, deduplicating only identical full rows.

    All physical source line numbers survive in ``source_rows``. Contradictory
    alarm definitions remain distinct definitions, not resolved records.
    """

    path = Path(path).resolve()
    source_sha256 = _sha256(path)
    source_rows: dict[tuple[str, str, str, bool], list[int]] = {}
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError("state dictionary must have the four expected columns in order")
        for row in reader:
            count += 1
            if None in row or any(row[column] is None for column in REQUIRED_COLUMNS):
                raise ValueError(f"malformed state dictionary row at line {reader.line_num}")
            sensor_type, state_set_id, state_text, alarm_raw = (
                row[column] for column in REQUIRED_COLUMNS
            )
            if not sensor_type or not state_set_id or not state_text:
                raise ValueError(f"empty state dictionary key at line {reader.line_num}")
            if alarm_raw not in {"true", "false"}:
                raise ValueError(f"invalid state dictionary alarm at line {reader.line_num}")
            key = sensor_type, state_set_id, state_text, alarm_raw == "true"
            source_rows.setdefault(key, []).append(reader.line_num)
    definitions = tuple(
        StateDefinition(*key, tuple(lines))
        for key, lines in sorted(
            source_rows.items(),
            key=lambda item: (item[0][0], _set_sort_key(item[0][1]), item[0][2], item[0][3]),
        )
    )
    grouped: dict[tuple[str, str], list[StateDefinition]] = {}
    for definition in definitions:
        grouped.setdefault((definition.sensor_type, definition.state_text), []).append(definition)
    return StateDictionary(
        path=path,
        sha256=source_sha256,
        source_row_count=count,
        definitions=definitions,
        types=frozenset(item.sensor_type for item in definitions),
        by_key={key: tuple(values) for key, values in grouped.items()},
    )


def match_state(
    dictionary: StateDictionary,
    sensor_type: str | None,
    state_text: str,
    observed_alarm: bool,
) -> dict[str, Any]:
    """Return candidates without using observed alarm to choose among them."""

    if not isinstance(state_text, str) or not isinstance(observed_alarm, bool):
        raise ValueError("a text state and boolean observed alarm are required")
    match_key = state_text.strip()
    if sensor_type not in dictionary.types:
        definitions: tuple[StateDefinition, ...] = ()
        status = "unmapped_type"
    else:
        definitions = dictionary.by_key.get((sensor_type, match_key), ())
        if not definitions:
            status = "unmapped_state"
        else:
            alarms = {item.expected_alarm for item in definitions}
            sets = {item.state_set_id for item in definitions}
            if len(alarms) > 1:
                status = "conflicting_definition"
            elif len(sets) > 1:
                status = "multiple_candidates"
            else:
                status = "exact_candidate"
    set_ids = sorted({item.state_set_id for item in definitions}, key=_set_sort_key)
    expected_values = sorted({item.expected_alarm for item in definitions})
    expected = expected_values[0] if len(expected_values) == 1 else None
    consistency = (
        "undetermined"
        if expected is None
        else "agree"
        if expected == observed_alarm
        else "disagree"
    )
    return {
        "state_match_key": match_key,
        "match_status": status,
        "candidate_set_ids": set_ids,
        "expected_alarm_values": expected_values,
        "expected_alarm": expected,
        "alarm_consistency": consistency,
        "definition_source_rows": sorted(
            {line for item in definitions for line in item.source_rows}
        ),
    }


def build_audit_table(
    aggregates: Iterable[Mapping[str, Any]],
    dictionary: StateDictionary,
    *,
    input_manifest_sha256: str,
) -> pa.Table:
    """Map pre-aggregated clean text rows without multiplying their weights."""

    if (
        not isinstance(input_manifest_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", input_manifest_sha256) is None
    ):
        raise ValueError("input_manifest_sha256 must be a SHA-256 hex string")
    config = {"mapping_version": MAPPING_VERSION, "normalization_rule": NORMALIZATION_RULE}
    config_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_id = f"r1-state-{input_manifest_sha256[:12]}-{dictionary.sha256[:12]}"
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, str | None, str, bool]] = set()
    for aggregate in aggregates:
        year_value = aggregate["year"]
        sensor_type = aggregate["sensor_type"]
        state_raw = aggregate["state_text_raw"]
        observed_alarm = aggregate["observed_alarm"]
        row_count_value = aggregate["row_count"]
        if (
            isinstance(year_value, bool)
            or not isinstance(year_value, Integral)
            or isinstance(row_count_value, bool)
            or not isinstance(row_count_value, Integral)
        ):
            raise ValueError("aggregate year and row_count must be integers")
        year = int(year_value)
        row_count = int(row_count_value)
        if not isinstance(state_raw, str) or not isinstance(observed_alarm, bool):
            raise ValueError("aggregate must contain text state and boolean alarm")
        if not 1 <= row_count or not 1 <= year <= 9999:
            raise ValueError("invalid aggregate year or row_count")
        key = year, sensor_type, state_raw, observed_alarm
        if key in seen:
            raise ValueError(f"duplicate aggregate key {key!r}")
        seen.add(key)
        output.append(
            {
                "schema_version": MAPPING_VERSION,
                "run_id": run_id,
                "config_sha256": config_sha,
                "input_manifest_sha256": input_manifest_sha256,
                "dictionary_sha256": dictionary.sha256,
                "year": year,
                "sensor_type": sensor_type,
                "state_text_raw": state_raw,
                "observed_alarm": observed_alarm,
                "row_count": row_count,
                "normalization_rule": NORMALIZATION_RULE,
                **match_state(dictionary, sensor_type, state_raw, observed_alarm),
            }
        )
    output.sort(
        key=lambda row: (
            row["year"],
            row["sensor_type"] or "",
            row["state_text_raw"],
            row["observed_alarm"],
        )
    )
    table = pa.Table.from_pylist(output, schema=STATE_MAPPING_SCHEMA)
    if table.num_rows != len(seen) or sum(table.column("row_count").to_pylist()) != sum(
        int(row["row_count"]) for row in output
    ):
        raise AssertionError("state mapping unexpectedly changed aggregate row multiplicities")
    return table
