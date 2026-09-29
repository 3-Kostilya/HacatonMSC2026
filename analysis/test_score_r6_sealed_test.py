"""Synthetic boundary checks before the single real sealed-test run."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb
import pandas as pd

from analysis.score_r6_sealed_test import score_month
from ml.forecast.r6_rule import TERMS


class R6TestScoringBoundaryTest(unittest.TestCase):
    def test_scores_only_test_rows_and_checks_future_episode(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            at = pd.Timestamp("2026-01-01 12:00:00")
            candidates = pd.DataFrame({
                "channel_id": ["a", "b", "c"],
                "prediction_time": [at, at, at],
                "sensor_type": ["smoke"] * 3,
                "target": [1, 0, 0],
                "target_episode_id": ["episode-a", None, None],
                "label_available_at": [at + pd.Timedelta(hours=2), pd.NaT, pd.NaT],
                "split": ["test", "test", "validation"],
            })
            features = candidates[["channel_id", "prediction_time", "sensor_type"]].copy()
            for name in TERMS:
                features[name] = 0
            features.loc[0, "registered_fault_text_count_24h"] = 4
            candidate_path = root / "candidate.parquet"
            features_path = root / "features.parquet"
            candidates.to_parquet(candidate_path, index=False)
            features.to_parquet(features_path, index=False)
            with duckdb.connect(":memory:") as database:
                result = score_month(database, candidate_path, features_path, 7.1)
                self.assertEqual(len(result), 2)
                self.assertEqual(result.rule_score.tolist(), [8.0, 0.0])
                self.assertEqual(result.above_frozen_threshold.tolist(), [True, False])
                candidates.loc[0, "label_available_at"] = at - pd.Timedelta(hours=1)
                candidates.to_parquet(candidate_path, index=False)
                with self.assertRaisesRegex(ValueError, "invalid horizon"):
                    score_month(database, candidate_path, features_path, 7.1)


if __name__ == "__main__":
    unittest.main()
