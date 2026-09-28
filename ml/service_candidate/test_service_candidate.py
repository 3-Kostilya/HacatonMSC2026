"""Portable model contract checks; no user data or large model required."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from ml.service_candidate.features import engineered_input
from ml.service_candidate.loader import ResearchRiskModel, sha256, source_sha256
from ml.service_candidate.train import episode_weights, write_json


def example_features():
    names = [
        "sensor_type",
        "last_observation_age_seconds",
        "last_completed_episode_end_age_seconds",
    ]
    for family in (
        "event_count",
        "technical_message_count",
        "registered_fault_text_count",
        "normal_message_count",
        "environmental_alarm_count",
        "unknown_state_count",
        "state_transitions",
    ):
        names.extend(f"{family}_{hours}h" for hours in (1, 6, 24, 168))
    names.extend(f"unused_{n}" for n in range(121 - len(names)))
    values = {name: [0.0, 1.0, 2.0, 3.0] for name in names if name != "sensor_type"}
    values["sensor_type"] = ["Датчик дыма", "Датчик дыма", "Состояние фазы", "Состояние фазы"]
    return pd.DataFrame(values), names


class TestServiceCandidate(unittest.TestCase):
    def test_source_hash_is_portable_across_windows_checkout_line_endings(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "feature_source.py"
            source.write_bytes(b"a = 1\nb = 2\n")
            lf_hash = source_sha256(source)
            source.write_bytes(b"a = 1\r\nb = 2\r\n")
            self.assertEqual(source_sha256(source), lf_hash)
            source.write_bytes(b"a = 1\r\nb = 3\r\n")
            self.assertNotEqual(source_sha256(source), lf_hash)

    def test_feature_contract_ignores_future_fields_and_keeps_missing(self):
        frame, names = example_features()
        frame["target"] = [0, 0, 1, 1]
        frame["unused_0"] = [None, 1, 2, 3]
        matrix = engineered_input(frame, names)
        self.assertNotIn("target", matrix)
        self.assertEqual(matrix.loc[0, "unused_0"], -1)
        self.assertTrue(np.isfinite(matrix.select_dtypes(include="number").to_numpy()).all())
        self.assertGreater(len(matrix.columns), len(names))

    def test_equal_positive_episode_mass(self):
        frame = pd.DataFrame(
            {"target": [1, 1, 1, 0, 0], "target_episode_id": ["a", "a", "b", None, None]}
        )
        weight, classes, details = episode_weights(frame)
        self.assertAlmostEqual(weight[:2].sum(), weight[2])
        self.assertEqual(classes, [1.0, 2 / 3])
        self.assertEqual(details["positive_episodes"], 2)

    def test_loader_scores_only_eligible_and_never_emits_alerts(self):
        frame, names = example_features()
        matrix = engineered_input(frame, names)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = CatBoostClassifier(
                iterations=4,
                depth=2,
                verbose=False,
                allow_writing_files=False,
                cat_features=["sensor_type"],
                thread_count=1,
            )
            model.fit(matrix, [0, 1, 0, 1])
            model.save_model(str(directory / "model.cbm"))
            write_json(
                directory / "model_metadata.json",
                {
                    "model_sha256": sha256(directory / "model.cbm"),
                    "feature_transform_source_sha256": source_sha256(
                        Path(__file__).with_name("features.py")
                    ),
                    "base_feature_names": names,
                    "engineered_feature_names": matrix.columns.tolist(),
                    "production_approved": False,
                    "automatic_actions_allowed": False,
                },
            )
            loaded = ResearchRiskModel(directory)
            frame["admission_status"] = ["eligible", "unknown", "excluded", "eligible"]
            scored = loaded.score(frame)
            self.assertEqual(
                scored.prediction_status.tolist(),
                ["scored_research", "not_available", "not_available", "scored_research"],
            )
            self.assertTrue(np.isnan(scored.risk_score.iloc[1:3]).all())
            self.assertTrue(scored.warning.isna().all())
            self.assertAlmostEqual(
                scored.risk_score.iloc[0], float(model.predict_proba(matrix.iloc[:1])[0, 1])
            )
            with self.assertRaises(ValueError):
                loaded.score(frame.assign(target=[0, 0, 0, 0]))
            with self.assertRaises(ValueError):
                loaded.score(frame.drop(columns="admission_status"))


if __name__ == "__main__":
    unittest.main()
