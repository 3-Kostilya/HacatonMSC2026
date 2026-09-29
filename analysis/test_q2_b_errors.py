"""Regression checks for fixed-threshold Q2/B error attribution."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.audit_q2_b_errors import _cooldown_sources


class Q2BErrorAttributionTest(unittest.TestCase):
    def test_missed_high_scores_attribute_the_prior_emitted_warning(self) -> None:
        scores = pd.DataFrame({
            "target_episode_id": ["a", "b"],
            "channel_id": ["gas", "smoke"],
            "prediction_time": pd.to_datetime(["2025-01-01 01:00",
                                               "2025-01-02 01:00"]),
            "target": [1, 1],
            "score_linear121": [0.95, 0.98],
        })
        episodes = pd.DataFrame({"target_episode_id": ["a", "b"],
                                 "error_class": ["cooldown_suppressed"] * 2})
        emitted = pd.DataFrame({
            "channel_id": ["gas", "smoke"],
            "prediction_time": pd.to_datetime(["2025-01-01 00:00",
                                               "2025-01-02 00:00"]),
            "outcome": ["no_target_in_horizon", "matched_episode"],
        })
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scores.parquet"
            pq.write_table(pa.Table.from_pandas(scores, preserve_index=False), path)
            with duckdb.connect(":memory:") as db:
                result, counts = _cooldown_sources(db, [path], episodes,
                                                    emitted, 0.9)
        self.assertEqual(len(result), 2)
        self.assertEqual(counts, {"no_target_in_horizon": 1,
                                  "matched_episode": 1})
        self.assertEqual(result.hours_after_blocking_warning.tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
