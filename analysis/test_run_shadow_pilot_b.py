"""Exercise the file handoff, saved decisions and resumed cooldown."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.run_shadow_pilot_b import run


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


if __name__ == "__main__":
    unittest.main()
