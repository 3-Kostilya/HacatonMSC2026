"""Full R3 population and month-shard boundary checks."""

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.audit_r3_population import INTERVAL_SCHEMA, _ceil_hour, _count_interval
from analysis.build_r3_full_month import (
    _intervals_for_month,
    _month_bounds,
    _prediction_times,
)


class FullR3GridTests(unittest.TestCase):
    def test_explicit_normal_selects_only_current_and_future_whole_hours(self) -> None:
        at = datetime(2025, 6, 1, 10, 0)
        self.assertEqual(_ceil_hour(at), at)
        self.assertEqual(_ceil_hour(at + timedelta(microseconds=1)), at + timedelta(hours=1))

    def test_split_count_keeps_boundary_hours_separate(self) -> None:
        counts: Counter[str] = Counter()
        _count_interval(datetime(2024, 12, 31, 23), datetime(2025, 1, 1, 2), counts)
        self.assertEqual(counts, {"train": 1, "validation": 2})

    def test_month_clips_and_expands_intervals_without_future_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "intervals.parquet"
            pq.write_table(pa.Table.from_pylist([
                {"channel_id": "c", "archive_segment": 1,
                 "start_at": datetime(2025, 5, 31, 23),
                 "end_exclusive": datetime(2025, 6, 1, 2)},
                {"channel_id": "c", "archive_segment": 1,
                 "start_at": datetime(2025, 6, 1, 3),
                 "end_exclusive": datetime(2025, 6, 1, 5)},
            ], schema=INTERVAL_SCHEMA), path)
            intervals, count = _intervals_for_month(
                path, datetime(2025, 6, 1), datetime(2025, 7, 1)
            )
            self.assertEqual(count, 4)
            self.assertEqual(_prediction_times(intervals["c"]), [
                datetime(2025, 6, 1, 0), datetime(2025, 6, 1, 1),
                datetime(2025, 6, 1, 3), datetime(2025, 6, 1, 4),
            ])

    def test_excluded_2021_cannot_be_published_as_feature_month(self) -> None:
        with self.assertRaisesRegex(ValueError, "accepted R1 archive"):
            _month_bounds("2021-06")


if __name__ == "__main__":
    unittest.main()
