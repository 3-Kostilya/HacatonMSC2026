"""Fixture tests for the full-column DuckDB/Parquet logical comparator."""

from pathlib import Path
import tempfile
import unittest

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.verify_m1_db_parquet import (
    FingerprintMismatchError,
    compare_db_to_parquet,
)
from stage1.ingestion.sql import export_tables


def _make_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    database = root / "work's.duckdb"
    artifact = root / "artifact's"
    connection = duckdb.connect(str(database))
    try:
        connection.execute(
            """CREATE TABLE raw(
                   row_id BIGINT, channel_id VARCHAR, source VARCHAR, source_row BIGINT
               )"""
        )
        connection.execute(
            """INSERT INTO raw VALUES
               (1, 'alpha', 'first.csv', 2),
               (2, 'alpha', 'second.csv', 2),
               (3, 'bad', 'second.csv', 3)"""
        )
        connection.execute(
            """CREATE TABLE classified(
                   row_id BIGINT, channel_id VARCHAR, payload VARCHAR,
                   source VARCHAR, source_row BIGINT, first_row_id BIGINT,
                   disposition VARCHAR, tags VARCHAR[]
               )"""
        )
        connection.execute(
            """INSERT INTO classified VALUES
               (1, 'alpha', 'kept', 'first.csv', 2, 1, 'accepted', ['one']),
               (2, 'alpha', 'copy', 'second.csv', 2, 1, 'exact_duplicate', ['two']),
               (3, 'bad', NULL, 'second.csv', 3, 3, 'quarantine', [])"""
        )
        connection.execute(
            """CREATE TABLE clean(
                   channel_id VARCHAR, sensor_type VARCHAR, value_numeric DOUBLE,
                   value_state VARCHAR, alarm BOOLEAN, timestamp TIMESTAMP,
                   value_raw VARCHAR, quality_flags VARCHAR[], source VARCHAR,
                   source_row BIGINT, row_id BIGINT, "odd""name" VARCHAR,
                   year INTEGER, month INTEGER
               )"""
        )
        connection.execute(
            """INSERT INTO clean VALUES
               ('alpha', 'temperature', 1.5, NULL, false, '2020-01-02 00:00:00',
                '1.5', ['ok'], 'first.csv', 2, 1, 'quoted-a', 2020, 1),
               ('beta', 'state', NULL, 'on', true, '2020-01-01 00:00:00',
                'on', ['flag'], 'first.csv', 3, 4, 'quoted-b', 2020, 1),
               ('gamma', NULL, 'NaN'::DOUBLE, NULL, false, '2021-12-31 23:59:59',
                NULL, [], 'second.csv', 4, 5, NULL, 2021, 12)"""
        )
        export_tables(connection, artifact)
    finally:
        connection.close()
    return database, artifact


def _rewrite(path: Path, transform) -> None:
    table = pq.ParquetFile(path).read()
    pq.write_table(transform(table), path)


class VerifyM1DatabaseParquetTests(unittest.TestCase):
    def test_matching_artifact_is_order_independent_and_reports_every_column(self):
        with tempfile.TemporaryDirectory() as temporary:
            database, artifact = _make_fixture(Path(temporary))
            partition = artifact / "clean" / "year=2020" / "month=1" / "data_0.parquet"
            _rewrite(partition, lambda table: table.take(pa.array([1, 0])))

            report = compare_db_to_parquet(database, artifact, memory_limit="64MB")

            self.assertEqual(report["status"], "verified")
            self.assertEqual(report["algorithm"]["duckdb_version"], duckdb.__version__)
            self.assertTrue(report["algorithm"]["order_independent"])
            self.assertEqual(len(report["clean"]["partitions"]), 2)
            columns = [item["name"] for item in report["clean"]["schema"]["database"]]
            self.assertIn('odd"name', columns)
            self.assertNotIn("year", columns)
            self.assertNotIn("month", columns)
            january = report["clean"]["partitions"][0]
            self.assertEqual(january["database"]["rows"], 2)
            self.assertEqual(january["database"], january["parquet"])
            self.assertTrue(report["excluded_rows"]["match"])
            self.assertTrue(report["sensor_statistics"]["match"])

    def test_change_in_last_clean_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            database, artifact = _make_fixture(Path(temporary))
            partition = artifact / "clean" / "year=2020" / "month=1" / "data_0.parquet"

            def alter(table):
                index = table.schema.get_field_index('odd"name')
                return table.set_column(index, 'odd"name', pa.array(["changed", "quoted-a"]))

            _rewrite(partition, alter)
            with self.assertRaises(FingerprintMismatchError) as raised:
                compare_db_to_parquet(database, artifact, memory_limit="64MB")
            report = raised.exception.report
            self.assertEqual(report["status"], "mismatch")
            self.assertFalse(report["clean"]["match"])
            self.assertEqual(report["clean"]["partitions"][0]["reason"], "logical_content_mismatch")
            self.assertTrue(report["excluded_rows"]["match"])

    def test_excluded_row_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            database, artifact = _make_fixture(Path(temporary))
            path = artifact / "excluded_rows.parquet"

            def alter(table):
                index = table.schema.get_field_index("duplicate_of_source")
                values = table.column(index).to_pylist()
                values[0] = "tampered.csv"
                return table.set_column(index, "duplicate_of_source", pa.array(values))

            _rewrite(path, alter)
            with self.assertRaises(FingerprintMismatchError) as raised:
                compare_db_to_parquet(database, artifact, memory_limit="64MB")
            self.assertFalse(raised.exception.report["excluded_rows"]["match"])
            self.assertEqual(
                raised.exception.report["excluded_rows"]["reason"],
                "logical_content_mismatch",
            )

    def test_sensor_statistics_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            database, artifact = _make_fixture(Path(temporary))
            path = artifact / "sensor_statistics.parquet"

            def alter(table):
                index = table.schema.get_field_index("distinct_values")
                values = table.column(index).to_pylist()
                values[-1] += 1
                return table.set_column(index, "distinct_values", pa.array(values))

            _rewrite(path, alter)
            with self.assertRaises(FingerprintMismatchError) as raised:
                compare_db_to_parquet(database, artifact, memory_limit="64MB")
            self.assertFalse(raised.exception.report["sensor_statistics"]["match"])

    def test_missing_partition_and_schema_change_are_rejected_with_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            database, artifact = _make_fixture(Path(temporary))
            missing = artifact / "clean" / "year=2021" / "month=12" / "data_0.parquet"
            missing.unlink()
            with self.assertRaises(FingerprintMismatchError) as raised:
                compare_db_to_parquet(database, artifact, memory_limit="64MB")
            reasons = {
                item.get("reason") for item in raised.exception.report["clean"]["partitions"]
            }
            self.assertIn("missing_parquet_partition", reasons)

            # Restore a fresh artifact, then alter a physical type without changing names.
            database, artifact = _make_fixture(Path(temporary) / "second")
            path = artifact / "clean" / "year=2020" / "month=1" / "data_0.parquet"

            def alter_schema(table):
                index = table.schema.get_field_index('odd"name')
                return table.set_column(index, 'odd"name', pa.array([1, 2], type=pa.int64()))

            _rewrite(path, alter_schema)
            with self.assertRaises(FingerprintMismatchError) as raised:
                compare_db_to_parquet(database, artifact, memory_limit="64MB")
            self.assertFalse(raised.exception.report["clean"]["match"])
            self.assertNotEqual(
                raised.exception.report["clean"]["schema"]["database"],
                raised.exception.report["clean"]["schema"]["parquet"],
            )


if __name__ == "__main__":
    unittest.main()
