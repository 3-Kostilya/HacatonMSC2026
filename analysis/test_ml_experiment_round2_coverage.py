"""Independent admission guards do not infer future labels or physical uptime."""
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import duckdb
import pandas as pd

from analysis.ml_experiment_round2_coverage import guard_violations


class CoverageGuardTests(unittest.TestCase):
    def row(self):
        at = datetime(2025, 1, 2, 12)
        return dict(prediction_time=at, sensor_type="Датчик дыма", admission_status="unknown",
                    last_explicit_normal_at=at-timedelta(hours=1), admission_evidence_through=at,
                    blocking_qa_count_24h=0, ambiguous_seconds_24h=0,
                    quality_rows_24h=1, excluded_quality_count_24h=1,
                    last_conflict_at=at-timedelta(hours=2), last_hard_quality_at=None,
                    admission_reasons=["quality_exclusions_24h"], combined_status="eligible",
                    first_usable_at=at-timedelta(days=10), second_usable_at=at-timedelta(days=9),
                    availability_status="unknown", protected_ok=True)

    def count(self, rows):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"admission.parquet"
            frame = pd.DataFrame(rows)
            # Preserve timestamp type even when all hard-quality timestamps are missing.
            frame["last_hard_quality_at"] = pd.to_datetime(frame["last_hard_quality_at"])
            frame.to_parquet(path, index=False)
            with duckdb.connect() as db:
                return guard_violations(db, path)

    def test_later_unambiguous_normal_can_release_only_old_conflict(self):
        row = self.row()
        self.assertEqual(self.count([row]), 0)
        row["last_conflict_at"] = row["last_explicit_normal_at"]
        self.assertEqual(self.count([row]), 1)

    def test_current_ambiguity_and_hard_errors_stay_blocked(self):
        for changes in [dict(ambiguous_seconds_24h=1),
                        dict(last_hard_quality_at=datetime(2025,1,2,10)),
                        dict(blocking_qa_count_24h=1)]:
            self.assertEqual(self.count([{**self.row(), **changes}]), 1)

    def test_future_and_cross_segment_evidence_is_rejected(self):
        row = self.row()
        row["admission_evidence_through"] += timedelta(hours=1)
        self.assertEqual(self.count([row]), 1)
        row = self.row()
        row["first_usable_at"] = datetime(2020,12,31)
        self.assertEqual(self.count([row]), 1)

    def test_cold_start_needs_two_different_past_observations(self):
        row = self.row()
        row.update(admission_reasons=["insufficient_history"], quality_rows_24h=0,
                   excluded_quality_count_24h=0, last_conflict_at=None)
        self.assertEqual(self.count([row]), 0)
        row["second_usable_at"] = row["first_usable_at"]
        self.assertEqual(self.count([row]), 1)

    def test_label_change_cannot_affect_admission_guard(self):
        negative = {**self.row(), "target": 0}
        positive = {**self.row(), "target": 1}
        self.assertEqual(self.count([negative, positive]), 0)


if __name__ == "__main__":
    unittest.main()
