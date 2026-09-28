"""Regression checks for the independent A2 real-channel verifier."""

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.verify_a2_real20 import verify
from stage1.features import A2_SCHEMA, FEATURE_VERSION, FeatureEvent, build_hourly_rows


class VerifyA2Real20Tests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.m1 = self.root / "m1"
        self.a2 = self.root / "a2"
        clean = self.m1 / "clean" / "year=2025" / "month=6"
        clean.mkdir(parents=True)
        self.a2.mkdir()
        self.hour = datetime(2025, 6, 15, 12)
        self.event = self.hour - timedelta(minutes=30)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "channel_id": "c",
                        "timestamp": self.event,
                        "alarm": True,
                        "value_numeric": 10.0,
                        "value_state": None,
                        "quality_flags": [],
                    }
                ]
            ),
            clean / "data_0.parquet",
        )
        m1_manifest = self.m1 / "manifest.json"
        m1_manifest.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        self.manifest_hash = hashlib.sha256(m1_manifest.read_bytes()).hexdigest()
        (self.a2 / "manifest.json").write_text(
            json.dumps({"status": "complete", "schema_version": FEATURE_VERSION}),
            encoding="utf-8",
        )
        self.row = build_hourly_rows(
            [FeatureEvent("c", self.event, True, value_numeric=10.0, sensor_type="temperature")],
            "c",
            self.hour,
            self.hour + timedelta(hours=1),
        )[0]
        self.row.update(
            schema_version=FEATURE_VERSION,
            run_id="fixture",
            config_sha256="a" * 64,
            input_manifest_sha256=self.manifest_hash,
        )
        self._write_feature()

    def _write_feature(self) -> None:
        pq.write_table(
            pa.Table.from_pylist([self.row], schema=A2_SCHEMA), self.a2 / "features.parquet"
        )

    def test_verifies_independent_counts_and_median(self) -> None:
        report = verify(self.m1, self.a2, expected_channels=1)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["sampled_hours"], 1)
        self.assertEqual(report["checked_fields"], 20)

    def test_detects_published_count_tampering(self) -> None:
        self.row["event_count_1h"] = 2
        self._write_feature()
        with self.assertRaisesRegex(AssertionError, "event_count"):
            verify(self.m1, self.a2, expected_channels=1)


if __name__ == "__main__":
    unittest.main()
