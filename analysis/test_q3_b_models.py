"""Verify time purging, unknown handling and strict Q3 candidate selection."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.run_q3_b_models import fold_bounds, row_query, select_candidate


class Q3ModelProtocolTest(unittest.TestCase):
    def test_fold_query_excludes_boundary_future_unknown_and_unapproved_delta(self):
        times = pd.to_datetime(["2022-12-30 22:00", "2022-12-31 00:00",
                                "2023-01-01 00:00", "2022-12-30 23:00",
                                "2022-12-30 21:00"])
        features = pd.DataFrame({"channel_id": ["x"] * 5, "prediction_time": times,
                                 "sensor_type": ["Датчик дыма"] * 5, "value": [1] * 5})
        labels = features.drop(columns="value").assign(
            target=pd.Series([1, 0, 0, None, 1], dtype="Int64"),
            target_episode_id=["e", None, None, None, "e"],
            label_available_at=times + pd.Timedelta(hours=1),
            horizon_end=times + pd.Timedelta(hours=24), split_status="assigned")
        delta = pd.DataFrame({"channel_id": ["x"], "prediction_time": [times[4]],
                              "combined_status": ["unknown"]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            part = {"features": root / "base.parquet", "labels": root / "labels.parquet",
                    "delta_features": root / "extra.parquet", "delta_admission": root / "admission.parquet"}
            for key, frame in (("features", features.iloc[:4]), ("labels", labels),
                               ("delta_features", features.iloc[4:]), ("delta_admission", delta)):
                pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), part[key])
            with duckdb.connect() as db:
                result = db.execute(row_query(part, "combined", 2023, training=True)).fetch_df()
        self.assertEqual(result.prediction_time.tolist(), [times[0]])
        self.assertTrue(result.source_base.all())

    def test_strict_goals_and_diagnostic_only_fallback(self):
        candidate = {"policy": "base", "model": "base51", "threshold": 0.9,
                     "episode_precision": 0.7, "full_episode_recall": 0.5,
                     "full_episode_f1": 0.58, "unmatched_warnings_per_1000_channel_days": 1.0}
        decision = select_candidate([candidate])
        self.assertIsNone(decision["working_selection"])
        self.assertEqual(decision["candidate_status"], "diagnostic_only")
        better = {**candidate, "policy": "combined", "episode_precision": 0.71,
                  "full_episode_recall": 0.51}
        self.assertEqual(select_candidate([candidate, better])["working_selection"], better)

    def test_only_predeclared_internal_years(self):
        self.assertEqual(fold_bounds(2023, training=True)[1], pd.Timestamp("2023-01-01"))
        with self.assertRaises(ValueError):
            fold_bounds(2026, training=False)


if __name__ == "__main__":
    unittest.main()
