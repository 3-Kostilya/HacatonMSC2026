import json
import unittest

import numpy as np
import pandas as pd

from dataset import TARGET
from full_study import OUTPUT, STUDY, build_full, split_full
from predict import predict


class FullPeriodTests(unittest.TestCase):
    def test_split_boundaries_and_purge(self):
        dates = [
            "2019-01-01",
            "2023-12-30 23:00",
            "2023-12-31",
            "2024-01-01",
            "2024-12-31",
            "2025-01-01",
            "2026-06-29 23:00",
            "2026-06-30",
            "2026-07-01",
        ]
        frame = pd.DataFrame({"timestamp": pd.to_datetime(dates, format="mixed")})
        self.assertEqual(
            split_full(frame).split.tolist(),
            [
                "train",
                "train",
                "excluded",
                "validation",
                "excluded",
                "test",
                "test",
                "excluded",
                "excluded",
            ],
        )

    def test_history_is_preserved_across_year_boundary(self):
        events = pd.DataFrame(
            {
                "channel_id": "1",
                "timestamp": pd.date_range("2023-12-30", periods=6, freq="24h"),
                "state": ["Норма", "Норма", "Норма", "Неисправен", "Норма", "Норма"],
                "alarm": False,
            }
        )
        data, _, audit = build_full(events)
        row = data.loc[data.timestamp.eq("2024-01-01")].iloc[0]
        self.assertEqual(row.split, "validation")
        self.assertEqual(row[TARGET], 1)
        self.assertEqual(row.history_hours, 48)
        self.assertEqual(row.events_168h, 3)
        self.assertEqual(audit["labelled_hours"], len(data))


@unittest.skipUnless((OUTPUT / "model_metadata.json").exists(), "Run the full-period study first")
class FullModelTests(unittest.TestCase):
    def test_saved_model_and_temporal_scope(self):
        metadata = json.loads((OUTPUT / "model_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["training_years"], STUDY["training_years"])
        self.assertEqual(metadata["validation_period"], STUDY["validation_period"])
        self.assertEqual(metadata["test_period"], STUDY["test_period"])
        data = pd.read_parquet(OUTPUT / "training_dataset.parquet")
        self.assertTrue(
            (
                data.loc[data.split.eq("train"), "timestamp"] + pd.Timedelta(hours=24)
                < pd.Timestamp("2024-01-01")
            ).all()
        )
        self.assertTrue(
            (
                data.loc[data.split.eq("validation"), "timestamp"] + pd.Timedelta(hours=24)
                < pd.Timestamp("2025-01-01")
            ).all()
        )
        predictions = pd.read_parquet(OUTPUT / "test_predictions.parquet")
        events = pd.read_parquet(OUTPUT / "events.parquet")
        samples = pd.concat(
            [
                predictions.head(2),
                predictions.loc[predictions[TARGET].eq(1)].head(2),
                predictions.tail(2),
            ]
        ).drop_duplicates()
        for row in samples.itertuples():
            history = events.loc[events.channel_id.eq(row.channel_id)]
            actual = predict(history, row.timestamp, model_dir=OUTPUT)
            np.testing.assert_allclose(
                actual.risk_score.iloc[0], row.risk_score, rtol=0, atol=1e-12
            )
            truncated = predict(
                history.loc[history.timestamp.le(row.timestamp)], row.timestamp, model_dir=OUTPUT
            )
            pd.testing.assert_frame_equal(actual, truncated)


if __name__ == "__main__":
    unittest.main()
