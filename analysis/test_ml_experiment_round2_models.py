"""Train-weight semantics independent of model quality and hidden-year data."""
import unittest

import numpy as np
import pandas as pd

from analysis.ml_experiment_round2_models import weights_for


def fixture():
    onset = pd.Timestamp("2023-07-01")
    leads = [2, 12, 4, 5, 23, 0, 0]
    return pd.DataFrame({"target": [1, 1, 1, 1, 1, 0, 0],
                         "target_episode_id": ["a", "a", "b", "b", "b", None, None],
                         "prediction_time": [onset - pd.Timedelta(hours=lead) for lead in leads],
                         "label_available_at": [onset] * 5 + [pd.NaT, pd.NaT]})


class Round2ModelsTest(unittest.TestCase):
    def test_equal_episode_mass_and_unchanged_targets(self):
        frame = fixture()
        targets = frame.target.copy()
        weight, classes, detail = weights_for(frame, "episode")
        self.assertAlmostEqual(float(weight[:2].sum()), float(weight[2:5].sum()))
        self.assertTrue(frame.target.equals(targets))
        self.assertEqual(weight[5:].tolist(), [1, 1])
        self.assertEqual(detail["negative_rows_retained"], 2)
        self.assertAlmostEqual(classes[1], 2 / 5)

    def test_lead_kernel_redistributes_inside_episode_only(self):
        weight, _, detail = weights_for(fixture(), "episode_lead")
        self.assertAlmostEqual(float(weight[:2].sum()), float(weight[2:5].sum()))
        self.assertAlmostEqual(float(weight[0] / weight[1]), 3)
        self.assertAlmostEqual(float(weight[2] / weight[4]), 3, places=6)
        self.assertEqual(detail["emphasized_1_to_6h_rows"], 3)
        self.assertEqual(weight[5:].tolist(), [1, 1])

    def test_recent_weight_keeps_every_negative_and_positive(self):
        frame = fixture()
        frame.loc[5, "prediction_time"] = pd.Timestamp("2019-02-01")
        frame.loc[6, "prediction_time"] = pd.Timestamp("2022-02-01")
        weight, _, detail = weights_for(frame, "episode_recent")
        self.assertEqual(len(weight), len(frame))
        self.assertTrue(np.all(weight > 0))
        self.assertAlmostEqual(float(weight[5]), .25)
        self.assertAlmostEqual(float(weight[6]), .7, places=6)
        self.assertFalse(detail["targets_changed"])

    def test_invalid_positive_lead_is_rejected(self):
        frame = fixture()
        frame.loc[0, "prediction_time"] = frame.loc[0, "label_available_at"]
        with self.assertRaisesRegex(ValueError, "positive leadtime"):
            weights_for(frame, "episode_lead")
