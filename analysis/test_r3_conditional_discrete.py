"""Conditional admission keeps unknown channel continuity explicit."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_r3_conditional_discrete import _eligible_rows


class ConditionalDiscreteTests(unittest.TestCase):
    def test_admits_only_assigned_binary_labels_with_discrete_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels, statuses = root / "labels.parquet", root / "status.parquet"
            start = datetime(2025, 6, 1)
            label_rows = []
            status_rows = []
            scenarios = [
                (1, "positive", "assigned", "eligible", "unknown"),
                (0, "negative", "assigned", "eligible", "unknown"),
                (None, "unknown", "assigned", "eligible", "unknown"),
                (1, "positive", "purged_boundary", "eligible", "unknown"),
                (1, "positive", "assigned", "unknown", "unknown"),
                (1, "positive", "assigned", "eligible", "excluded"),
            ]
            for hour, (target, label, split_status, discrete, availability) in enumerate(scenarios):
                at = start + timedelta(hours=hour)
                label_rows.append({
                    "channel_id": "c", "prediction_time": at, "sensor_type": "Датчик дыма",
                    "target": target, "label_status": label, "split_status": split_status,
                    "split": "validation", "target_episode_id": "episode" if target == 1 else None,
                    "label_available_at": at + timedelta(hours=1),
                })
                status_rows.append({
                    "channel_id": "c", "prediction_time": at,
                    "discrete_data_status": discrete,
                    "availability_status": availability,
                })
            pq.write_table(pa.Table.from_pylist(label_rows), labels)
            pq.write_table(pa.Table.from_pylist(status_rows), statuses)
            with duckdb.connect(":memory:") as database:
                rows = [row for batch in _eligible_rows(database, labels, statuses)
                        for row in batch.to_pylist()]
            self.assertEqual([row["target"] for row in rows], [1, 0])
            self.assertEqual({row["availability_status"] for row in rows}, {"unknown"})
            self.assertEqual(
                {row["admission_status"] for row in rows},
                {"conditional_archive_assumption"},
            )


if __name__ == "__main__":
    unittest.main()
