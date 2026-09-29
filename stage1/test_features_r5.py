"""Causality and out-of-sample contracts for R5 A anomaly features."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from stage1.features.r5 import (
    INPUT_COLUMNS, STAT_COLUMNS, fit_type_models, score_type_models,
    statistical_features,
)


def _rows(n: int, *, start: datetime = datetime(2020, 1, 1)) -> pd.DataFrame:
    records = []
    for i in range(n):
        row = {name: i % 5 for name in INPUT_COLUMNS}
        row.update({
            "channel_id": f"c{i % 5}",
            "prediction_time": start + timedelta(hours=i),
            "sensor_type": "Датчик дыма",
            "event_count_1h": i % 3,
            "event_count_6h": i % 7 + 2,
            "event_count_24h": i % 11 + 12,
            "event_count_168h": i % 31 + 90,
            "alarm_count_1h": i % 2,
            "alarm_count_24h": i % 5 + 5,
            "state_transitions_6h": i % 3,
            "state_transitions_168h": i % 17 + 10,
        })
        records.append(row)
    return pd.DataFrame.from_records(records)


class R5FeaturesTests(unittest.TestCase):
    def test_b_ablation_contract_excludes_diagnostics_and_target(self) -> None:
        contract = json.loads((Path(__file__).resolve().parents[1] / "ml" /
                               "r5_a_feature_contract_v1.json").read_text(encoding="utf-8"))
        groups = contract["feature_groups"]
        names = [name for group in groups.values() for name in group]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(set(STAT_COLUMNS) <= set(names))
        self.assertFalse(set(names) & set(contract["diagnostic_columns_not_model_features"]))
        self.assertFalse(set(names) & {"target", "target_episode_id", "split"})
        r3 = json.loads((Path(__file__).resolve().parents[1] / "ml" /
                         "r3_discrete_feature_allowlist_v1.json").read_text(encoding="utf-8"))
        self.assertTrue(set(INPUT_COLUMNS) <= set(r3["feature_names"]))

    def test_statistics_use_only_current_past_window_values(self) -> None:
        frame = _rows(5)
        first = statistical_features(frame).iloc[0]
        frame.loc[4, "event_count_1h"] = 100000
        self.assertEqual(first.to_dict(), statistical_features(frame).iloc[0].to_dict())
        self.assertEqual(set(first.index), set(STAT_COLUMNS))
        self.assertTrue(np.isfinite(first.to_numpy()).all())

    def test_fit_rejects_future_and_scores_out_of_sample(self) -> None:
        train = _rows(100)
        cutoff = pd.Timestamp("2020-02-01")
        model = fit_type_models(train, fit_end_at=cutoff)
        self.assertIsNotNone(model)
        assert model is not None
        scored = score_type_models(_rows(4, start=datetime(2020, 2, 1)), model)
        self.assertEqual(len(scored), 4)
        self.assertTrue(np.isfinite(scored["r5_if_anomaly_score"]).all())
        self.assertTrue(np.isfinite(scored["r5_kmeans_distance"]).all())
        self.assertEqual(model.fit_channels, 5)
        extended = score_type_models(_rows(5, start=datetime(2020, 2, 1)), model)
        for name in (*STAT_COLUMNS, "r5_if_anomaly_score", "r5_kmeans_distance"):
            np.testing.assert_array_equal(scored[name].to_numpy(), extended[name].to_numpy()[:4])
        with self.assertRaisesRegex(ValueError, "future"):
            fit_type_models(train, fit_end_at=pd.Timestamp("2020-01-02"))
        with self.assertRaisesRegex(ValueError, "precedes"):
            score_type_models(train, model)

    def test_sparse_type_keeps_stats_but_null_models(self) -> None:
        frame = _rows(3)
        self.assertIsNone(fit_type_models(frame, fit_end_at=pd.Timestamp("2021-01-01")))
        scored = score_type_models(frame, None)
        self.assertTrue(scored["r5_if_anomaly_score"].isna().all())
        self.assertTrue(scored[list(STAT_COLUMNS)].notna().all().all())


if __name__ == "__main__":
    unittest.main()
