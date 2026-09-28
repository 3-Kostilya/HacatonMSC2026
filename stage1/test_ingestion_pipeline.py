"""End-to-end contracts for milestone 1, including cross-file/chunk cases."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import duckdb
import pyarrow.parquet as pq

from stage1.ingestion.pipeline import run_ingestion
from stage1.ingestion.pipeline import parquet_sanity
from stage1.ingestion.sql import export_tables
from stage1.ingestion.dictionaries import CHANNEL_REQUIRED_COLUMNS, OBJECT_REQUIRED_COLUMNS
from stage1.normalization import EVENT_FIELDS


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


class IngestionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.channels = self.root / "channels.csv"
        self.objects = self.root / "objects.csv"
        write_csv(
            self.channels,
            [*CHANNEL_REQUIRED_COLUMNS, "ид_объект"],
            [
                ["c1", "sys", "Датчик температуры", "tag", "one", "o1"],
                ["c2", "sys", "Газовый датчик", "tag", "two", "absent"],
            ],
        )
        write_csv(self.objects, OBJECT_REQUIRED_COLUMNS, [["o1", "1", "", "building", "Building"]])

    def config(self, files, name="out", batch_size=1):
        return {
            "sources": [{"path": str(p), "max_rows": None} for p in files],
            "channels": str(self.channels),
            "objects": str(self.objects),
            "output": str(self.root / name),
            "memory_limit": "128MB",
            "batch_size": batch_size,
        }

    def test_global_dedup_join_preservation_and_partitioned_values(self):
        a = ["id1", "c1", "2026-01-02", "00:00:00", "t", "-3276"]
        b = ["id1", "c1", "2026-01-01", "00:00:00", "false", "327.68"]
        x, y = self.root / "a.csv", self.root / "b.csv"
        write_csv(
            x,
            EVENT_FIELDS,
            [
                a,
                list(EVENT_FIELDS),
                b,
                ["id5", "c1", "broken", "00:00:00", "f", "keep me"],
                ["id3", "unknown", "2026-02-01", "00:00:00", "false", "Неисправен"],
            ],
        )
        write_csv(
            y,
            EVENT_FIELDS,
            [
                a,
                ["id2", "c1", "2026-01-02", "00:00:00", "true", "999"],
                ["id4", "c2", "2026-02-02", "00:00:00", "true", "12,5"],
                ["id6", "c1", "2026-01-03", "00:00:00", "f", "NaN"],
            ],
        )
        config = self.config([x, y])
        report = run_ingestion(config)
        self.assertEqual(report["input_rows"], 9)
        self.assertEqual(
            report["dispositions"],
            {"accepted": 6, "quarantine": 1, "repeated_header": 1, "exact_duplicate": 1},
        )
        self.assertTrue(all(report["sanity_checks"].values()))
        self.assertEqual(report["repeated_event_ids"], 1)
        files = sorted((self.root / "out/clean").rglob("*.parquet"))
        self.assertEqual(len(files), 2)
        records = [r for p in files for r in pq.ParquetFile(p).read().to_pylist()]
        numeric = {r["value_numeric"] for r in records if r["is_numeric"]}
        self.assertEqual(numeric, {-3276, 327.68, 999, 12.5})
        self.assertEqual(sum("channel_time_conflict" in r["quality_flags"] for r in records), 2)
        self.assertEqual(
            next(r for r in records if r["channel_id"] == "unknown")["join_status"],
            "unknown_channel",
        )
        self.assertEqual(
            next(r for r in records if r["channel_id"] == "c2")["join_status"], "object_not_found"
        )
        self.assertEqual(next(r for r in records if r["value_raw"] == "NaN")["value_state"], "NaN")
        excluded = pq.read_table(self.root / "out/excluded_rows.parquet").to_pylist()
        self.assertEqual(
            next(r for r in excluded if r["disposition"] == "quarantine")["value_raw"], "keep me"
        )
        duplicate = next(r for r in excluded if r["disposition"] == "exact_duplicate")
        self.assertEqual(duplicate["duplicate_of_source"], str(x.resolve()))
        self.assertFalse((self.root / "out/work.duckdb").exists())
        with self.assertRaises(FileExistsError):
            run_ingestion(config)

    def test_empty_input_and_missing_mapping_are_explicit(self):
        write_csv(self.channels, CHANNEL_REQUIRED_COLUMNS, [["c1", "sys", "type", "tag", "one"]])
        source = self.root / "empty.csv"
        write_csv(source, EVENT_FIELDS, [])
        report = run_ingestion(self.config([source]))
        self.assertEqual(report["input_rows"], 0)
        self.assertFalse(report["dictionary_audit"]["object_mapping_available"])
        self.assertTrue(all(report["sanity_checks"].values()))

    def test_batch_size_does_not_change_output(self):
        source = self.root / "rows.csv"
        write_csv(source, EVENT_FIELDS, [["id", "c1", "2026-01-01", "00:00", "t", "1"]] * 5)
        reports = [
            run_ingestion(self.config([source], name=f"batch{n}", batch_size=n)) for n in (1, 3)
        ]
        self.assertEqual(reports[0]["dispositions"], reports[1]["dispositions"])
        self.assertEqual(reports[0]["dispositions"], {"accepted": 1, "exact_duplicate": 4})

    def test_failure_is_not_published_as_complete(self):
        source = self.root / "bad.csv"
        source.write_text("bad,header\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            run_ingestion(self.config([source]))
        manifest = json.loads(
            (self.root / "out.inprogress/manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse((self.root / "out").exists())

    def test_missing_object_cell_differs_from_missing_mapping_column(self):
        source = self.root / "rows.csv"
        write_csv(source, EVENT_FIELDS, [["id", "c1", "2026-01-01", "00:00", "t", "1"]])
        for has_column in (True, False):
            fields = (
                [*CHANNEL_REQUIRED_COLUMNS, "ид_объект"] if has_column else CHANNEL_REQUIRED_COLUMNS
            )
            row = ["c1", "sys", "type", "tag", "one"] + ([""] if has_column else [])
            write_csv(self.channels, fields, [row])
            report = run_ingestion(self.config([source], name=f"mapping{has_column}"))
            expected = "object_id_missing" if has_column else "object_mapping_unavailable"
            self.assertEqual(report["join_status"], [{"join_status": expected, "rows": 1}])

    def test_failed_sanity_does_not_publish_parquet(self):
        from unittest.mock import patch

        source = self.root / "rows.csv"
        write_csv(source, EVENT_FIELDS, [["id", "c1", "2026-01-01", "00:00", "t", "1"]])
        with patch("stage1.ingestion.pipeline.parquet_sanity", return_value={"sorted": False}):
            with self.assertRaisesRegex(ValueError, "sanity checks failed"):
                run_ingestion(self.config([source]))
        self.assertFalse((self.root / "out").exists())
        self.assertTrue((self.root / "out.inprogress/clean").exists())

    def test_multiple_row_groups_remain_sorted_within_each_month(self):
        # More than one row group and interleaved source order: small fixtures
        # cannot expose the partitioned-writer ordering regression.
        output = self.root / "large_export"
        output.mkdir()
        with duckdb.connect() as con:
            con.execute("SET threads=1")
            con.execute("""CREATE TABLE clean AS SELECT
                i AS row_id, (i % 127)::VARCHAR AS channel_id,
                TIMESTAMP '2026-01-01' + i * INTERVAL '1 second' AS timestamp,
                'type' AS sensor_type, i::DOUBLE AS value_numeric,
                NULL::VARCHAR AS value_state, i::VARCHAR AS value_raw,
                false AS alarm, 2026 AS year, (1+i%2)::INTEGER AS month
                FROM range(300000) t(i)""")
            con.execute("""CREATE TABLE raw (
                row_id BIGINT, source VARCHAR, source_row BIGINT)""")
            con.execute("""CREATE TABLE classified (
                row_id BIGINT, first_row_id BIGINT, disposition VARCHAR)""")
            export_tables(con, output)
        self.assertTrue(all(parquet_sanity(output, 300000).values()))
        files = list((output / "clean").rglob("*.parquet"))
        self.assertEqual(len(files), 2)
        self.assertTrue(all(pq.ParquetFile(p).metadata.num_row_groups > 1 for p in files))


if __name__ == "__main__":
    unittest.main()
