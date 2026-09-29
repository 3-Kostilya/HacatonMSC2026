"""Full B3 ingestion must preserve every A3 key in causal channel order."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_r3_full_registered_labels import _prediction_groups


class FullB3JoinTests(unittest.TestCase):
    def _paths(self, root: Path, *, wrong_status_key: bool = False,
               unsorted: bool = False) -> tuple[Path, Path]:
        at = datetime(2025, 6, 1)
        points = [("a", at), ("a", at + timedelta(hours=1)), ("b", at)]
        if unsorted:
            points[1], points[2] = points[2], points[1]
        features = root / "features.parquet"
        statuses = root / "row_status.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"channel_id": channel, "prediction_time": time,
             "sensor_type": "Датчик дыма"} for channel, time in points
        ]), features)
        if wrong_status_key:
            points[0] = ("c", at)
        pq.write_table(pa.Table.from_pylist([
            {"channel_id": channel, "prediction_time": time,
             "availability_status": "unknown", "numeric_data_status": "unknown",
             "discrete_data_status": "unknown"} for channel, time in points
        ]), statuses)
        return features, statuses

    def test_groups_all_keys_by_channel(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            groups = list(_prediction_groups(*self._paths(Path(temp))))
            self.assertEqual([(channel, len(rows)) for channel, rows in groups],
                             [("a", 2), ("b", 1)])

    def test_rejects_key_mismatch_or_unsorted_channels(self) -> None:
        for kwargs, error in (({"wrong_status_key": True}, "keys differ"),
                              ({"unsorted": True}, "not sorted")):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError, error):
                    list(_prediction_groups(*self._paths(Path(temp), **kwargs)))


if __name__ == "__main__":
    unittest.main()
