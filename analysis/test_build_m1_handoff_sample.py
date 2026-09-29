"""Contract tests for the published-M1 handoff fragment."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_m1_handoff_sample import build_sample


class HandoffSampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = self.root / "published"
        clean = self.artifact / "clean" / "year=2025" / "month=1"
        clean.mkdir(parents=True)
        self.rows = [
            (1, "temperature", 21.0, None, "object_mapping_unavailable", [], False),
            (
                2,
                "temperature",
                31.0,
                None,
                "object_mapping_unavailable",
                ["channel_time_conflict"],
                True,
            ),
            (3, "smoke", None, "normal", "object_mapping_unavailable", [], False),
            (4, "smoke", None, "alarm", "object_mapping_unavailable", [], True),
            (5, None, None, "on", "unknown_channel", ["unknown_channel"], False),
            (6, None, 2.0, None, "unknown_channel", ["unknown_channel"], False),
        ]
        self.schema = pa.schema(
            [
                ("row_id", pa.int64()),
                ("source", pa.string()),
                ("source_row", pa.int64()),
                ("channel_id", pa.string()),
                ("timestamp", pa.timestamp("us")),
                ("sensor_type", pa.string()),
                ("value_numeric", pa.float64()),
                ("value_state", pa.string()),
                ("join_status", pa.string()),
                ("quality_flags", pa.list_(pa.string())),
                ("alarm", pa.bool_()),
            ]
        )
        records = [
            dict(
                row_id=row_id,
                source="events.csv",
                source_row=row_id + 1,
                channel_id=str(row_id),
                timestamp=datetime(2025, 1, 1, row_id),
                sensor_type=sensor_type,
                value_numeric=numeric,
                value_state=state,
                join_status=join,
                quality_flags=flags,
                alarm=alarm,
            )
            for row_id, sensor_type, numeric, state, join, flags, alarm in self.rows
        ]
        pq.write_table(pa.Table.from_pylist(records, schema=self.schema), clean / "data_0.parquet")
        (self.artifact / "manifest.json").write_text(
            json.dumps({"status": "complete", "scope": "full_supplied_sources"}),
            encoding="utf-8",
        )

    def test_all_categories_and_schema_are_preserved_deterministically(self) -> None:
        first_path = self.root / "first.parquet"
        second_path = self.root / "second.parquet"
        first = build_sample(self.artifact, first_path, max_rows=32, buckets=3)
        second = build_sample(self.artifact, second_path, max_rows=32, buckets=3)
        self.assertEqual(first["sample_sha256"], second["sample_sha256"])
        self.assertEqual(first["source_manifest_sha256"], second["source_manifest_sha256"])
        self.assertEqual(first["missing_categories"], [])
        self.assertIn("join_status:unknown_channel", first["covered_categories"])
        self.assertIn("quality_flag:channel_time_conflict", first["covered_categories"])
        self.assertIn("sensor_type_value_kind:temperature:numeric", first["covered_categories"])
        table = pq.read_table(first_path)
        self.assertTrue(table.schema.equals(self.schema, check_metadata=False))
        self.assertEqual(table.num_rows, 6)
        self.assertEqual(set(table.column("source").to_pylist()), {"events.csv"})
        self.assertEqual(first["counts_by_join_status"]["unknown_channel"], 2)
        self.assertEqual(
            json.loads(first_path.with_suffix(".report.json").read_text(encoding="utf-8"))[
                "sample_sha256"
            ],
            first["sample_sha256"],
        )

    def test_budget_limit_explicitly_reports_lost_coverage(self) -> None:
        result = build_sample(self.artifact, self.root / "one.parquet", max_rows=1)
        self.assertEqual(result["sample_rows"], 1)
        self.assertTrue(result["missing_categories"])
        self.assertEqual(
            sorted(result["covered_categories"] + result["missing_categories"]),
            result["observed_categories"],
        )

    def test_refuses_unpublished_or_unsafe_destination(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            build_sample(self.artifact, self.artifact / "sample.parquet")
        (self.artifact / "manifest.json").write_text(
            json.dumps({"status": "running", "scope": "full_supplied_sources"}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "not complete"):
            build_sample(self.artifact, self.root / "bad.parquet")
        self.artifact.rename(self.root / "published.inprogress")
        with self.assertRaisesRegex(ValueError, "inprogress"):
            build_sample(self.root / "published.inprogress", self.root / "bad.parquet")

    def test_bounded_requires_explicit_override(self) -> None:
        (self.artifact / "manifest.json").write_text(
            json.dumps({"status": "complete", "scope": "bounded_probe"}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "full-history"):
            build_sample(self.artifact, self.root / "bad.parquet")
        result = build_sample(self.artifact, self.root / "pilot.parquet", allow_bounded=True)
        self.assertEqual(result["source_scope"], "bounded_probe")


if __name__ == "__main__":
    unittest.main()
