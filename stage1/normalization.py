"""Bounded-memory normalization of the stage-one event journal.

The input journals are much larger than RAM.  ``normalize_chunks`` therefore
spools only a narrow typed representation to a temporary SQLite database.  It
uses that disk-backed index to identify duplicates and channel/time conflicts
across chunk boundaries, then yields records one at a time.
"""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from stage1.contracts import NormalizedEvent


EVENT_FIELDS = (
    "ид_события",
    "ид_канала_данных",
    "дата",
    "время",
    "тревожное",
    "значение_датчика",
)

TRUE_ALARMS = frozenset({"true", "t", "1"})
FALSE_ALARMS = frozenset({"false", "f", "0"})

UNKNOWN_CHANNEL = "unknown_channel"
MISSING_EVENT_ID = "missing_event_id"
EMPTY_CHANNEL_ID = "empty_channel_id"
EMPTY_VALUE = "empty_value"
INVALID_ALARM = "invalid_alarm"
INVALID_TIMESTAMP = "invalid_timestamp"
TIMEZONE_PRESENT = "timezone_present"
NONFINITE_NUMERIC = "nonfinite_numeric"
CHANNEL_TIME_CONFLICT = "channel_time_conflict"
EXACT_DUPLICATE = "exact_duplicate"


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """A traceable normalization result; rejected rows have no event."""

    source: str
    source_row: int
    disposition: str
    event: NormalizedEvent | None
    quality_flags: tuple[str, ...]
    duplicate_of_source: str | None = None
    duplicate_of_source_row: int | None = None

    def to_record(self) -> dict[str, Any]:
        event = self.event
        return {
            "source": self.source,
            "source_row": self.source_row,
            "disposition": self.disposition,
            "duplicate_of_source": self.duplicate_of_source,
            "duplicate_of_source_row": self.duplicate_of_source_row,
            "event_id": event.event_id if event else None,
            "channel_id": event.channel_id if event else None,
            "timestamp": event.timestamp if event else None,
            "sensor_type": event.sensor_type if event else None,
            # raw_value is the lossless text representation; numeric_value is
            # parsed independently and never replaces it.
            "raw_value": event.raw_value if event else None,
            "numeric_value": event.numeric_value if event else None,
            "alarm": event.alarm if event else None,
            "object_id": event.object_id if event else None,
            "quality_flags": list(self.quality_flags),
        }


def parse_alarm(value: Any) -> bool:
    """Parse only the explicitly supported journal alarm spellings."""

    normalized = str(value).strip().casefold()
    if normalized in TRUE_ALARMS:
        return True
    if normalized in FALSE_ALARMS:
        return False
    raise ValueError(INVALID_ALARM)


def parse_local_timestamp(date_value: Any, time_value: Any) -> datetime:
    """Parse journal date/time and reject timezone-aware timestamps."""

    text = f"{str(date_value).strip()} {str(time_value).strip()}"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(INVALID_TIMESTAMP) from error
    if stamp.tzinfo is not None and stamp.utcoffset() is not None:
        raise ValueError(TIMEZONE_PRESENT)
    return stamp


def parse_numeric(value: str) -> tuple[float | None, str | None]:
    """Return an optional finite number while keeping the original text."""

    try:
        number = float(value.strip().replace(",", "."))
    except ValueError:
        return None, None
    if not math.isfinite(number):
        return None, NONFINITE_NUMERIC
    return number, None


def _canonical_key(values: Iterable[str]) -> str:
    """Serialize field boundaries losslessly; unlike a hash, this is exact."""

    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _raw_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field, "")
    return "" if value is None else str(value)


def _prepare_row(
    row: Mapping[str, Any], channel_types: Mapping[str, str], source: str
) -> tuple[NormalizedEvent | None, tuple[str, ...], str | None]:
    event_id = _raw_text(row, "ид_события").strip()
    channel_id = _raw_text(row, "ид_канала_данных").strip()
    raw_value = _raw_text(row, "значение_датчика")
    flags: list[str] = []
    if not event_id:
        flags.append(MISSING_EVENT_ID)
    if not channel_id:
        flags.append(EMPTY_CHANNEL_ID)
    if not raw_value.strip():
        flags.append(EMPTY_VALUE)
    try:
        timestamp = parse_local_timestamp(row.get("дата", ""), row.get("время", ""))
    except ValueError as error:
        timestamp = None
        flags.append(str(error))
    try:
        alarm = parse_alarm(row.get("тревожное", ""))
    except ValueError:
        alarm = None
        flags.append(INVALID_ALARM)

    sensor_type = channel_types.get(channel_id)
    if sensor_type is None and channel_id:
        sensor_type = "unknown"
        flags.append(UNKNOWN_CHANNEL)
    numeric_value, numeric_flag = parse_numeric(raw_value)
    if numeric_flag:
        flags.append(numeric_flag)

    fatal = {EMPTY_CHANNEL_ID, EMPTY_VALUE, INVALID_ALARM, INVALID_TIMESTAMP, TIMEZONE_PRESENT}
    if fatal.intersection(flags):
        return None, tuple(flags), None
    assert timestamp is not None and alarm is not None and sensor_type is not None
    event = NormalizedEvent(
        channel_id=channel_id,
        timestamp=timestamp,
        raw_value=raw_value,
        alarm=alarm,
        sensor_type=sensor_type,
        source=source,
        event_id=event_id or None,
        numeric_value=numeric_value,
        quality_flags=tuple(flags),
    )
    semantic_hash = _canonical_key((raw_value, "1" if alarm else "0"))
    return event, tuple(flags), semantic_hash


def normalize_chunks(
    chunks: Iterable[Iterable[Mapping[str, Any]]],
    channel_types: Mapping[str, str],
    source: str,
    *,
    first_source_row: int = 2,
    temp_directory: Path | None = None,
) -> Iterator[NormalizationResult]:
    """Normalize chunks with global duplicate/conflict checks in bounded RAM.

    An exact duplicate is a later row whose six raw source fields are byte-for-
    text identical after CSV decoding.  ``event_id`` alone is deliberately not
    a key.  A channel/time conflict means two valid, non-duplicate semantic
    payloads (alarm plus raw value) share a channel and timestamp; every member
    is retained and flagged instead of choosing one.
    """

    if not source.strip():
        raise ValueError("source must be a non-empty string")
    if first_source_row < 1:
        raise ValueError("first_source_row must be positive")

    def generate() -> Iterator[NormalizationResult]:
        with tempfile.NamedTemporaryFile(
            prefix="stage1-normalization-",
            suffix=".sqlite3",
            dir=temp_directory,
            delete=False,
        ) as temporary:
            database_path = Path(temporary.name)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(database_path)
            connection.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                CREATE TABLE rows (
                    seq INTEGER PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    source_row INTEGER NOT NULL,
                    exact_hash TEXT NOT NULL,
                    channel_id TEXT,
                    timestamp TEXT,
                    event_id TEXT,
                    sensor_type TEXT,
                    raw_value TEXT,
                    numeric_value REAL,
                    alarm INTEGER,
                    flags TEXT NOT NULL,
                    semantic_hash TEXT
                );
                CREATE INDEX exact_idx ON rows(exact_hash, seq);
                CREATE INDEX conflict_idx
                    ON rows(channel_id, timestamp, semantic_hash);
                """
            )
            seq = 0
            fallback_source_row = first_source_row
            for chunk in chunks:
                batch = []
                for row in chunk:
                    row_source = _raw_text(row, "__source__").strip() or source
                    explicit_source_row = row.get("__source_row__")
                    source_row = (
                        int(explicit_source_row)
                        if explicit_source_row is not None
                        else fallback_source_row
                    )
                    raw_fields = tuple(_raw_text(row, field) for field in EVENT_FIELDS)
                    event, flags, semantic_hash = _prepare_row(row, channel_types, row_source)
                    batch.append(
                        (
                            seq,
                            row_source,
                            source_row,
                            _canonical_key(raw_fields),
                            event.channel_id if event else None,
                            event.timestamp.isoformat(sep=" ") if event else None,
                            event.event_id if event else None,
                            event.sensor_type if event else None,
                            event.raw_value if event else None,
                            event.numeric_value if event else None,
                            int(event.alarm) if event else None,
                            json.dumps(flags, ensure_ascii=False),
                            semantic_hash,
                        )
                    )
                    seq += 1
                    fallback_source_row += 1
                connection.executemany(
                    "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
                )
                connection.commit()

            cursor = connection.execute(
                """
                WITH exact AS (
                    SELECT exact_hash, MIN(seq) AS first_seq
                    FROM rows GROUP BY exact_hash
                ), conflicts AS (
                    SELECT channel_id, timestamp
                    FROM rows
                    WHERE channel_id IS NOT NULL AND timestamp IS NOT NULL
                    GROUP BY channel_id, timestamp
                    HAVING COUNT(DISTINCT semantic_hash) > 1
                )
                SELECT r.source_name, r.source_row, r.channel_id, r.timestamp,
                       r.event_id, r.sensor_type,
                       r.raw_value, r.numeric_value, r.alarm, r.flags,
                       e.first_seq, first.source_name, first.source_row,
                       CASE WHEN c.channel_id IS NULL THEN 0 ELSE 1 END AS conflict,
                       r.seq
                FROM rows r
                JOIN exact e ON e.exact_hash = r.exact_hash
                JOIN rows first ON first.seq = e.first_seq
                LEFT JOIN conflicts c
                  ON c.channel_id = r.channel_id AND c.timestamp = r.timestamp
                ORDER BY r.seq
                """
            )
            for stored in cursor:
                (
                    row_source,
                    row_number,
                    channel_id,
                    timestamp_text,
                    event_id,
                    sensor_type,
                    raw_value,
                    numeric_value,
                    alarm_int,
                    flags_json,
                    first_seq,
                    first_source,
                    first_row,
                    conflict,
                    row_seq,
                ) = stored
                flags = list(json.loads(flags_json))
                duplicate = row_seq != first_seq
                if duplicate:
                    flags.append(EXACT_DUPLICATE)
                if conflict:
                    flags.append(CHANNEL_TIME_CONFLICT)
                event = None
                if channel_id is not None:
                    event = NormalizedEvent(
                        channel_id=channel_id,
                        timestamp=datetime.fromisoformat(timestamp_text),
                        raw_value=raw_value,
                        alarm=bool(alarm_int),
                        sensor_type=sensor_type,
                        source=row_source,
                        event_id=event_id,
                        numeric_value=numeric_value,
                        quality_flags=tuple(flags),
                    )
                disposition = (
                    "rejected" if event is None else (EXACT_DUPLICATE if duplicate else "accepted")
                )
                yield NormalizationResult(
                    source=row_source,
                    source_row=row_number,
                    disposition=disposition,
                    event=event,
                    quality_flags=tuple(flags),
                    duplicate_of_source=first_source if duplicate else None,
                    duplicate_of_source_row=first_row if duplicate else None,
                )
        finally:
            if connection is not None:
                connection.close()
            database_path.unlink(missing_ok=True)

    return generate()


def iter_accepted(results: Iterable[NormalizationResult]) -> Iterator[NormalizedEvent]:
    """Yield valid, first-occurrence events; conflicts remain visibly flagged."""

    for result in results:
        if result.disposition == "accepted" and result.event is not None:
            yield result.event
