"""Checks that M1 verification rejects incomplete or inconsistent handoffs."""

import json
import hashlib
from datetime import datetime
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.verify_m1_artifact import verify_artifact


class VerifyM1ArtifactTests(unittest.TestCase):
    def make_artifact(self, directory: Path, *, scope="full_supplied_sources") -> None:
        partition = directory / "clean" / "year=2020" / "month=1"
        partition.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "channel_id": ["a", "b"],
                    "timestamp": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                    "row_id": [1, 2],
                    "source": ["fixture", "fixture"],
                    "source_row": [2, 3],
                    "sensor_type": ["fixture", "fixture"],
                    "join_status": ["unknown", "unknown"],
                }
            ),
            partition / "data_0.parquet",
        )
        pq.write_table(
            pa.table({"source": pa.array([], type=pa.string())}),
            directory / "excluded_rows.parquet",
        )
        pq.write_table(
            pa.table({"channel_id": ["a", "b"]}), directory / "sensor_statistics.parquet"
        )
        parquet_bytes = sum(path.stat().st_size for path in directory.rglob("*.parquet"))
        source_path = directory / "fixture.csv"
        source_path.write_bytes(b"fixture")
        sources = [
            {
                "path": str(source_path),
                "bytes": source_path.stat().st_size,
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "max_rows": None,
                "rows_read": 2,
            }
        ]
        manifest = {
            "status": "complete",
            "scope": scope,
            "input_rows": 2,
            "sources": sources,
            "sanity_checks": {"row_conservation": True},
        }
        report = {
            "scope": scope,
            "input_rows": 2,
            "sources": sources,
            "dispositions": {"accepted": 2},
            "sanity_checks": {"joins_do_not_multiply_rows": True},
            "by_partition": [{"year": 2020, "month": 1, "rows": 2}],
            "by_type": [{"sensor_type": "fixture", "rows": 2}],
            "join_status": [{"join_status": "unknown", "rows": 2}],
            "parquet_bytes": parquet_bytes,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (directory / "data_quality.json").write_text(json.dumps(report), encoding="utf-8")

    def test_complete_full_artifact_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "full"
            self.make_artifact(directory)
            result = verify_artifact(
                directory, expected_source_paths={directory / "fixture.csv"}, deep=True
            )
            self.assertEqual((result["input_rows"], result["accepted_rows"]), (2, 2))
            self.assertEqual(result["status"], "verified_deep")

    def test_bounded_and_inprogress_are_not_full_handoffs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "probe"
            self.make_artifact(directory, scope="bounded_probe")
            with self.assertRaisesRegex(ValueError, "bounded probe"):
                verify_artifact(directory)
            self.assertEqual(
                verify_artifact(directory, require_full=False)["status"], "verified_sampled"
            )
            with self.assertRaisesRegex(ValueError, "inprogress"):
                verify_artifact(directory.with_name("probe.inprogress"))

    def test_wrong_partition_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "full"
            self.make_artifact(directory)
            path = directory / "data_quality.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["by_partition"][0]["rows"] = 3
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "partition row counts"):
                verify_artifact(directory, expected_source_paths={directory / "fixture.csv"})

    def test_missing_source_or_wrong_partition_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "full"
            self.make_artifact(directory)
            source = directory / "fixture.csv"
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source SHA-256 differs"):
                verify_artifact(directory, expected_source_paths={source})
            source.write_bytes(b"fixture")
            path = directory / "clean" / "year=2020" / "month=1" / "data_0.parquet"
            table = pq.read_table(path)
            index = table.schema.get_field_index("timestamp")
            table = table.set_column(
                index, "timestamp", pa.array([datetime(2020, 1, 1), datetime(2021, 1, 1)])
            )
            pq.write_table(table, path)
            with self.assertRaisesRegex(ValueError, "wrong year/month"):
                verify_artifact(directory, expected_source_paths={source})

    def test_wrong_inventory_and_fake_type_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "full"
            self.make_artifact(directory)
            source = directory / "fixture.csv"
            with self.assertRaisesRegex(ValueError, "expected source inventory"):
                verify_artifact(directory, expected_source_paths={directory / "different.csv"})
            path = directory / "data_quality.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["by_type"][0]["sensor_type"] = "fabricated"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "real clean type counts"):
                verify_artifact(directory, expected_source_paths={source}, deep=True)


if __name__ == "__main__":
    unittest.main()
