import unittest

import numpy as np
import pandas as pd

from extract import OUT
from predict import predict
from train import choose_threshold


class ThresholdTests(unittest.TestCase):
    def test_threshold_uses_precision_constraint(self):
        threshold, rule = choose_threshold([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.9])
        self.assertAlmostEqual(threshold, 0.8)
        self.assertIn("precision > 0.7", rule)

    def test_fallback_when_no_precision_threshold_exists(self):
        threshold, rule = choose_threshold([0, 0, 1], [0.9, 0.8, 0.1])
        self.assertAlmostEqual(threshold, 0.1)
        self.assertIn("unattainable", rule)


@unittest.skipUnless(
    (OUT / "events.parquet").exists() and (OUT / "model_metadata.json").exists(),
    "Run training before the saved-model integration tests",
)
class SavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = pd.read_parquet(OUT / "events.parquet")
        cls.test = pd.read_parquet(OUT / "test_predictions.parquet")

    def test_inference_matches_held_out_feature_pipeline(self):
        for row in self.test.groupby("channel_id").head(1).itertuples():
            events = self.events.loc[self.events.channel_id.eq(row.channel_id)].copy()
            actual = predict(events, row.timestamp)
            self.assertEqual(actual.status.iloc[0], "scored")
            np.testing.assert_allclose(
                actual.risk_score.iloc[0], row.risk_score, atol=1e-12, rtol=0
            )

    def test_future_events_do_not_change_inference(self):
        row = self.test.loc[self.test.sensor_fault_onset_24h.eq(1)].iloc[0]
        events = self.events.loc[self.events.channel_id.eq(row.channel_id)].copy()
        full = predict(events, row.timestamp)
        truncated = predict(events.loc[events.timestamp.le(row.timestamp)], row.timestamp)
        pd.testing.assert_frame_equal(full, truncated)

    def test_current_fault_is_not_scored(self):
        frame = pd.DataFrame(
            {
                "channel_id": [str(self.test.channel_id.iloc[0])] * 2,
                "timestamp": pd.to_datetime(["2023-08-01", "2023-08-03"]),
                "state": ["Норма", "Неисправен"],
                "alarm": [False, True],
            }
        )
        result = predict(frame, "2023-08-03")
        self.assertEqual(result.status.iloc[0], "already_faulty")
        self.assertTrue(pd.isna(result.risk_score.iloc[0]))

    def test_long_silence_is_not_scored(self):
        frame = pd.DataFrame(
            {
                "channel_id": [str(self.test.channel_id.iloc[0])] * 2,
                "timestamp": pd.to_datetime(["2023-08-01", "2023-08-03"]),
                "state": ["Норма", "Норма"],
                "alarm": [False, False],
            }
        )
        result = predict(frame, "2023-08-05")
        self.assertEqual(result.status.iloc[0], "stale_observation")
        self.assertTrue(pd.isna(result.risk_score.iloc[0]))


if __name__ == "__main__":
    unittest.main()
