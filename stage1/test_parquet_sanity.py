"""Focused physical-order checks for the vectorized Parquet verifier."""

from datetime import datetime
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from stage1.ingestion.pipeline import parquet_sanity


class ParquetSanityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        (self.output / "clean/year=2026/month=1").mkdir(parents=True)

    def write_keys(self, channel_ids, timestamps, row_ids):
        table = pa.table(
            {
                "channel_id": channel_ids,
                "timestamp": timestamps,
                "row_id": row_ids,
            }
        )
        pq.write_table(table, self.output / "clean/year=2026/month=1/data_0.parquet")

    def test_detects_disorder_inside_a_batch(self):
        moment = datetime(2026, 1, 1)
        self.write_keys(["a", "c", "b"], [moment] * 3, [1, 2, 3])

        checks = parquet_sanity(self.output, 3)

        self.assertTrue(checks["parquet_row_count_matches"])
        self.assertFalse(checks["parquet_files_sorted"])

    def test_detects_disorder_at_batch_boundary(self):
        moment = datetime(2026, 1, 1)
        rows = 25001
        self.write_keys(["a"] * rows, [moment] * rows, [*range(25000), -1])

        checks = parquet_sanity(self.output, rows)

        self.assertTrue(checks["parquet_row_count_matches"])
        self.assertFalse(checks["parquet_files_sorted"])


if __name__ == "__main__":
    unittest.main()
