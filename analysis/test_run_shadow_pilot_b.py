"""Exercise the file handoff, saved decisions and resumed cooldown."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.run_shadow_pilot_b import run
from analysis.train_r4_discrete_baselines import sha256
from ml.forecast.shadow_pilot import PINNED_FREEZE_SHA256


FREEZE = Path("ml/r6_frozen_rule_v1.json")


def source_row(at: datetime, *, admission: str = "eligible") -> dict:
    return {
        "channel_id": "smoke-1",
        "sensor_type": "Датчик дыма",
        "prediction_time": at,
        "admission_status": admission,
        "admission_reason": "coverage_unverified" if admission != "eligible" else None,
        "history_through": at,
        "admission_through": at,
        "registered_fault_text_count_24h": 3,
        "registered_fault_text_count_168h": 2,
        "completed_episode_count_168h": 0,
        "technical_message_count_24h": 1,
    }


class ShadowBatchTest(unittest.TestCase):
    def test_writes_record_only_output_and_resumes_cooldown(self) -> None:
        at = datetime(2026, 6, 30, 23)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_input = root / "first.parquet"
            pq.write_table(pa.Table.from_pylist([
                source_row(at), source_row(at + timedelta(hours=1), admission="unknown"),
            ]), first_input)
            first = run(input_path=first_input, freeze_path=FREEZE,
                        output_dir=root / "first_output", expected_channel_hours=2)
            self.assertEqual(first["summary"]["shadow_warnings"], 1)
            self.assertEqual(first["summary"]["unavailable_hours"], 1)

            second_input = root / "second.parquet"
            pq.write_table(pa.Table.from_pylist([
                source_row(at + timedelta(hours=23)),
                source_row(at + timedelta(hours=24)),
            ]), second_input)
            second = run(input_path=second_input, freeze_path=FREEZE,
                         checkpoint_in=root / "first_output" / "checkpoint.json",
                         output_dir=root / "second_output", expected_channel_hours=2)
            self.assertEqual(second["summary"]["shadow_warnings"], 1)
            decisions = pq.read_table(root / "second_output" /
                                      "shadow_decisions.parquet").to_pylist()
            self.assertEqual([item["warning_reason"] for item in decisions],
                             ["channel_cooldown", "recorded_shadow_warning"])
            self.assertTrue(all(not item["automatic_action_taken"] for item in decisions))

    def test_future_label_column_is_refused_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "bad.parquet"
            pq.write_table(pa.Table.from_pylist([
                {**source_row(datetime(2026, 6, 1)), "target": 1}
            ]), input_path)
            with self.assertRaisesRegex(ValueError, "future_fields"):
                run(input_path=input_path, freeze_path=FREEZE,
                    output_dir=root / "bad_output")
            self.assertFalse((root / "bad_output").exists())

    def test_a_package_hash_and_freeze_are_checked_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "hours.parquet"
            pq.write_table(pa.Table.from_pylist([
                source_row(datetime(2026, 6, 1)),
            ]), input_path)
            report_path = root / "report.json"
            report_path.write_text(json.dumps({
                "source_freeze_lf_sha256": PINNED_FREEZE_SHA256,
            }), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest = {
                "schema_version": "shadow-pilot-a-causal-admission-v1",
                "b_input_file": input_path.name,
                "b_input_sha256": sha256(input_path),
                "report_sha256": sha256(report_path),
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_path = root / "valid_output"
            result = run(input_path=input_path, freeze_path=FREEZE,
                         output_dir=output_path, source_manifest=manifest_path)
            self.assertEqual(result["source_a_manifest_sha256"], sha256(manifest_path))
            self.assertGreaterEqual(result["resources"]["elapsed_seconds"], 0)
            self.assertGreater(result["resources"]["peak_working_set_bytes"], 0)

            manifest["b_input_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source package differs"):
                run(input_path=input_path, freeze_path=FREEZE,
                    output_dir=root / "invalid_output",
                    source_manifest=manifest_path)
            self.assertFalse((root / "invalid_output").exists())


if __name__ == "__main__":
    unittest.main()
