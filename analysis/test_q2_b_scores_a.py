"""Full-score acceptance rejects changed keys, labels and serialized values."""

import unittest

import duckdb
import numpy as np
import pandas as pd

from analysis.verify_q2_b_scores_a import check_lineage, compare_scores


class ScoreAcceptanceTests(unittest.TestCase):
    def test_comparison_uses_identical_float32_serialization(self):
        source = np.array([0.1, 0.5, 0.9993761875927455], dtype=np.float64)
        result = compare_scores(source, source.astype(np.float32))
        self.assertEqual(result["different_float32_values"], 0)
        changed = source.astype(np.float32)
        changed[1] = np.nextafter(changed[1], np.float32(1))
        self.assertEqual(compare_scores(source, changed)["different_float32_values"], 1)

    def test_invalid_shape_nonfinite_and_range_rejected(self):
        for repeated, stored in (([0.5], []), ([np.nan], [0.5]), ([0.5], [np.inf]), ([1.1], [0.5])):
            with self.subTest(repeated=repeated, stored=stored), self.assertRaises(ValueError):
                compare_scores(repeated, stored)

    def test_full_outer_lineage_rejects_drop_extra_duplicate_and_wrong_label(self):
        expected = pd.DataFrame(
            {
                "channel_id": ["a", "b"],
                "prediction_time": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "sensor_type": ["Датчик дыма"] * 2,
                "label_sensor_type": ["Датчик дыма"] * 2,
                "target": [0, 1],
                "target_episode_id": [None, "b-one"],
                "label_available_at": pd.to_datetime(
                    ["2025-01-02", "2025-01-02 01:00"], format="mixed"
                ),
            }
        )
        scores = expected.drop(columns="label_sensor_type").copy()
        for name in ("base51", "full121", "linear121"):
            scores["score_" + name] = [0.1, 0.9]
        with duckdb.connect() as db:
            db.register("expected", expected)
            db.register("scores", scores)
            self.assertEqual(check_lineage(db)["joined_rows"], 2)
            variants = [scores.iloc[:1], pd.concat([scores, scores.iloc[[0]]], ignore_index=True)]
            extra = scores.copy()
            extra.loc[0, "channel_id"] = "extra"
            variants.append(extra)
            bad_label = scores.copy()
            bad_label.loc[0, "target"] = 1
            variants.append(bad_label)
            bad_score = scores.copy()
            bad_score.loc[0, "score_full121"] = np.nan
            variants.append(bad_score)
            for frame in variants:
                db.register("scores", frame)
                with self.assertRaises(ValueError):
                    check_lineage(db)


if __name__ == "__main__":
    unittest.main()
