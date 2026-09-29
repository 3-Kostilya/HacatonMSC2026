"""Tests for the fixed-population R5 B comparison boundary."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb
import pandas as pd

from analysis.run_r5_b_ablation import joined_month, model_input, variants
from analysis.train_r4_discrete_baselines import read_json


class R5BContractTest(unittest.TestCase):
    def test_variants_keep_mode_ids_out_of_model_features(self) -> None:
        contract = read_json(Path("ml/r5_a_feature_contract_v1.json"))
        groups = variants(contract)
        self.assertEqual(len(groups["full"]), 9)
        self.assertEqual(groups["baseline"], [])
        self.assertEqual(groups["change_point"], ["r5_event_change_z_6h"])
        self.assertFalse(any("mode_id" in name for name in groups["full"]))

    def test_missing_r5_score_keeps_training_row(self) -> None:
        frame = pd.DataFrame({
            "sensor_type": ["gas", "gas"],
            "registered_fault_text_count_24h": [1, 0],
            "r5_if_anomaly_score": [None, 0.25],
        })
        ready = model_input(frame, list(frame.columns))
        self.assertEqual(len(ready), 2)
        self.assertEqual(ready["r5_if_anomaly_score"].tolist(), [-1.0, 0.25])

    def test_join_preserves_same_validation_keys_and_type(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            a3, index, r5 = (root / name for name in ("a3", "index", "r5"))
            for folder in (a3, index, r5):
                folder.mkdir()
            times = pd.date_range("2025-01-01", periods=2, freq="h")
            keys = pd.DataFrame({
                "channel_id": ["a", "a"], "prediction_time": times,
                "sensor_type": ["gas", "gas"], "target": [1, 0],
                "target_episode_id": ["e1", None],
                "label_available_at": [times[0] + pd.Timedelta(hours=2), pd.NaT],
                "split": ["validation", "validation"],
            })
            features = keys[["channel_id", "prediction_time", "sensor_type"]].copy()
            features["registered_fault_text_count_24h"] = [2, 0]
            additions = keys[["channel_id", "prediction_time", "sensor_type"]].copy()
            additions["r5_if_anomaly_score"] = [0.4, 0.2]
            keys.to_parquet(index / "keys.parquet")
            features.to_parquet(a3 / "features.parquet")
            additions.to_parquet(r5 / "features.parquet")
            a = {"month": "2025-01", "features_file": "features.parquet"}
            i = {"manifest_file": "manifest.json"}
            (index / "conditional_discrete_keys.parquet").write_bytes(
                (index / "keys.parquet").read_bytes())
            r = {"features_file": "features.parquet"}
            with duckdb.connect(":memory:") as database:
                frame = joined_month(
                    database, a3, index, r5, a, i, r,
                    ["sensor_type", "registered_fault_text_count_24h"],
                    ["r5_if_anomaly_score"], "validation")
                self.assertEqual(len(frame), 2)
                self.assertEqual(frame["target"].tolist(), [1, 0])
                additions.loc[1, "sensor_type"] = "other"
                additions.to_parquet(r5 / "features.parquet")
                with self.assertRaisesRegex(ValueError, "join differs"):
                    joined_month(
                        database, a3, index, r5, a, i, r,
                        ["sensor_type", "registered_fault_text_count_24h"],
                        ["r5_if_anomaly_score"], "validation")

    def test_sealed_test_cannot_be_joined(self) -> None:
        with duckdb.connect(":memory:") as database:
            with self.assertRaisesRegex(ValueError, "train or validation"):
                joined_month(database, Path("."), Path("."), Path("."),
                             {}, {}, {}, [], [], "test")


if __name__ == "__main__":
    unittest.main()
