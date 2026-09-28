"""Q1/B must keep label keys and sensor types aligned before model fitting."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.run_q1_b_qa_ablation import active_qa_features, joined_month
from ml.forecast.v2_threshold import choose_threshold


T = datetime(2025, 1, 1)


def files(root: Path, *, feature_type: str = "Датчик дыма",
          duplicate_feature: bool = False) -> dict:
    labels = pd.DataFrame([
        {"channel_id": "c", "prediction_time": T, "sensor_type": "Датчик дыма",
         "target": 1, "target_episode_id": "e", "label_available_at": T + timedelta(hours=3),
         "split": "validation"},
        {"channel_id": "c", "prediction_time": T + timedelta(hours=1),
         "sensor_type": "Датчик дыма", "target": 0, "target_episode_id": None,
         "label_available_at": None, "split": "validation"},
    ])
    features = pd.DataFrame([
        {"channel_id": "c", "prediction_time": T, "sensor_type": feature_type,
         "qa_gas_count": 0},
        {"channel_id": "c", "prediction_time": T + timedelta(hours=1),
         "sensor_type": "Датчик дыма", "qa_gas_count": 1},
    ])
    if duplicate_feature:
        features = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    candidate = root / "candidate.parquet"
    feature_file = root / "features.parquet"
    pq.write_table(pa.Table.from_pandas(labels, preserve_index=False), candidate)
    pq.write_table(pa.Table.from_pandas(features, preserve_index=False), feature_file)
    return {"month": "2025-01", "split": "validation", "rows": 2,
            "candidate": candidate, "features": feature_file}


class Q1JoinTests(unittest.TestCase):
    def test_valid_join_preserves_all_validation_labels(self) -> None:
        with TemporaryDirectory() as directory, duckdb.connect(":memory:") as database:
            result = joined_month(database, files(Path(directory)),
                                  ["sensor_type", "qa_gas_count"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result.target.tolist(), [1, 0])
        self.assertEqual(result.target_episode_id.iloc[0], "e")

    def test_type_mismatch_and_duplicate_key_block_training(self) -> None:
        for kwargs in ({"feature_type": "ИБП"}, {"duplicate_feature": True}):
            with self.subTest(kwargs=kwargs), TemporaryDirectory() as directory:
                with duckdb.connect(":memory:") as database:
                    with self.assertRaisesRegex(ValueError, "keys, type or binary labels"):
                        joined_month(database, files(Path(directory), **kwargs),
                                     ["sensor_type", "qa_gas_count"])

    def test_qa_selection_uses_train_variation_only(self) -> None:
        train = pd.DataFrame({"qa_constant": [0, 0, 0], "qa_variable": [0, 1, 0]})
        self.assertEqual(active_qa_features(train, ["qa_constant", "qa_variable"]),
                         ["qa_variable"])

    def test_full_episode_goal_cannot_be_met_by_conditional_coverage(self) -> None:
        perfect_conditional = [{
            "threshold": 0.5, "matched_episodes": 261, "emitted_warnings": 261,
            "eligible_positive_episodes": 261, "episode_precision": 1.0,
            "episode_recall": 1.0,
        }]
        result = choose_threshold(perfect_conditional, full_positive_episodes=2142)
        self.assertIsNone(result["selected"])
        self.assertFalse(result["requirements_feasible_on_checked_grid"])


if __name__ == "__main__":
    unittest.main()
