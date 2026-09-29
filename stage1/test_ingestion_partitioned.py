"""Differential checks for the optional bounded-window SQL classifier."""

import csv
from pathlib import Path
import tempfile
import unittest

import duckdb
import pyarrow as pa

from stage1.ingestion.normalize import normalize_row
from stage1.ingestion.schemas import CHANNEL_SCHEMA, OBJECT_SCHEMA, STAGING_SCHEMA
from stage1.ingestion.sql import classify
from stage1.ingestion.sql_partitioned import classify_partitioned
from stage1.normalization import EVENT_FIELDS


def _rows():
    fixture = Path(__file__).parent / "fixtures" / "m0" / "events.csv"
    with fixture.open(encoding="utf-8", newline="") as stream:
        events = list(csv.DictReader(stream))
    # The first two source files include the same event.  Later files cross
    # calendar-year boundaries; each duplicate must still refer to the first
    # globally assigned row_id, not to the first row of its year or source.
    older = dict(zip(EVENT_FIELDS, ["old", "numeric", "2019-12-31", "23:00:00", "false", "1"]))
    newer = dict(zip(EVENT_FIELDS, ["new", "numeric", "2020-01-01", "00:00:00", "false", "2"]))
    repeated_header = dict(zip(EVENT_FIELDS, EVENT_FIELDS))
    invalid = dict(zip(EVENT_FIELDS, ["bad", "numeric", "broken", "00:00:00", "false", "3"]))
    sources = [
        events[:16] + [older, newer, repeated_header, invalid],
        events[16:] + [events[0], older, newer],
    ]
    normalized = []
    for source_number, source_rows in enumerate(sources):
        for source_row, event in enumerate(source_rows, start=2):
            normalized.append(
                normalize_row(
                    {
                        **event,
                        "__source__": f"source_{source_number}",
                        "__source_row__": source_row,
                    },
                    len(normalized) + 1,
                )
            )
    # SQL accepts nullable raw keys, even though the CSV parser normally
    # supplies strings.  Verify that null-equal keys hash and dedupe together.
    for source_number in (2, 3):
        row = normalize_row(
            {**older, "__source__": f"source_{source_number}", "__source_row__": 2},
            len(normalized) + 1,
        )
        row["event_id_raw"] = None
        normalized.append(row)
    return normalized


def _connection(rows):
    connection = duckdb.connect()
    for name, schema, records in (
        ("raw", STAGING_SCHEMA, rows),
        (
            "channels",
            CHANNEL_SCHEMA,
            [
                {
                    "channel_id": "numeric",
                    "sensor_type": "temperature",
                    "engineering_system_type": "hvac",
                    "engineering_system_tag": "tag",
                    "sensor_name": "sensor",
                    "object_id": "obj",
                }
            ],
        ),
        (
            "objects",
            OBJECT_SCHEMA,
            [
                {
                    "object_id": "obj",
                    "hierarchy_level": "1",
                    "parent_object_id": None,
                    "object_kind": "building",
                    "object_name": "building",
                }
            ],
        ),
    ):
        connection.register("incoming", pa.Table.from_pylist(records, schema=schema))
        connection.execute(f"CREATE TABLE {name} AS SELECT * FROM incoming")
        connection.unregister("incoming")
    return connection


class PartitionedClassifierTests(unittest.TestCase):
    def test_large_duplicate_skew_keeps_global_first_row(self):
        event = dict(zip(EVENT_FIELDS, ["same", "numeric", "2020-01-01", "00:00", "false", "1"]))
        rows = [
            normalize_row(
                {
                    **event,
                    "__source__": f"source_{index // 5000}",
                    "__source_row__": index % 5000 + 2,
                },
                index + 1,
            )
            for index in range(10000)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            candidate = _connection(rows)
            try:
                candidate.execute("SET memory_limit='64MB'")
                classify_partitioned(
                    candidate,
                    Path(temporary) / "scratch",
                    bucket_count=17,
                    writer_fanout=3,
                )
                self.assertEqual(
                    candidate.execute(
                        "SELECT disposition, count(*), min(first_row_id), max(first_row_id) "
                        "FROM classified GROUP BY disposition ORDER BY disposition"
                    ).fetchall(),
                    [("accepted", 1, 1, 1), ("exact_duplicate", 9999, 1, 1)],
                )
            finally:
                candidate.close()

    def test_empty_input_matches_global_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference = _connection([])
            candidate = _connection([])
            try:
                classify(reference, object_mapping_available=False)
                classify_partitioned(
                    candidate,
                    Path(temporary) / "scratch",
                    object_mapping_available=False,
                    bucket_count=3,
                    writer_fanout=2,
                )
                for table in ("classified", "conflicts", "clean"):
                    self.assertEqual(
                        reference.execute(f"DESCRIBE {table}").fetchall(),
                        candidate.execute(f"DESCRIBE {table}").fetchall(),
                    )
                    self.assertEqual(
                        reference.execute(f"SELECT count(*) FROM {table}").fetchone(),
                        candidate.execute(f"SELECT count(*) FROM {table}").fetchone(),
                    )
            finally:
                reference.close()
                candidate.close()

    def test_same_tables_as_global_window(self):
        rows = _rows()
        with tempfile.TemporaryDirectory() as temporary:
            reference = _connection(rows)
            candidate = _connection(rows)
            try:
                classify(reference)
                classify_partitioned(
                    candidate,
                    Path(temporary) / "scratch",
                    bucket_count=17,
                    writer_fanout=3,
                )
                for table, ordering in (
                    ("classified", "row_id"),
                    ("conflicts", "channel_id, timestamp"),
                    ("clean", "row_id"),
                ):
                    self.assertEqual(
                        reference.execute(f"DESCRIBE {table}").fetchall(),
                        candidate.execute(f"DESCRIBE {table}").fetchall(),
                        table,
                    )
                    self.assertEqual(
                        reference.execute(f"SELECT * FROM {table} ORDER BY {ordering}").fetchall(),
                        candidate.execute(f"SELECT * FROM {table} ORDER BY {ordering}").fetchall(),
                        table,
                    )
                firsts = dict(
                    candidate.execute(
                        "SELECT row_id, first_row_id FROM classified WHERE disposition='exact_duplicate'"
                    ).fetchall()
                )
                self.assertEqual(firsts[37], 1)  # First fixture event copied into source 2.
                self.assertEqual(firsts[38], 17)  # 2019 event copied into source 2.
                self.assertEqual(firsts[39], 18)  # 2020 event copied into source 2.
                self.assertEqual(firsts[41], 40)  # Nullable raw key copied into source 4.
            finally:
                reference.close()
                candidate.close()

    def test_rejects_existing_scratch_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scratch"
            path.mkdir()
            marker = path / "marker"
            marker.write_text("keep", encoding="utf-8")
            connection = _connection([])
            try:
                with self.assertRaises(FileExistsError):
                    classify_partitioned(connection, path)
                self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
