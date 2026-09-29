"""Tests for key-aligned R4 batch/replay comparison."""

from datetime import datetime, timedelta
import unittest

import pandas as pd

from analysis.audit_r4_a_batch_replay import compare_frames


def predictions() -> pd.DataFrame:
    at = datetime(2025, 1, 1)
    return pd.DataFrame({
        "channel_id": ["A", "B"],
        "prediction_time": [at, at + timedelta(hours=1)],
        "target": [1, 0],
        "rule_score": [7.1, 0.1],
        "logistic_score": [0.7, 0.2],
        "catboost_score": [0.8, 0.3],
    })


class ReplayComparisonTest(unittest.TestCase):
    def test_reordered_rows_and_roundoff_match(self) -> None:
        left = predictions()
        right = left.iloc[::-1].reset_index(drop=True)
        right.loc[0, "logistic_score"] += 1e-15
        result = compare_frames(left, right)
        self.assertFalse(result["source_row_order_equal"])
        self.assertEqual(result["rows"], 2)

    def test_target_difference_is_rejected(self) -> None:
        left = predictions()
        right = left.copy()
        right.loc[0, "target"] = 0
        with self.assertRaisesRegex(ValueError, "target"):
            compare_frames(left, right)

    def test_material_score_difference_is_rejected(self) -> None:
        left = predictions()
        right = left.copy()
        right.loc[0, "rule_score"] += 0.01
        with self.assertRaisesRegex(ValueError, "rule_score"):
            compare_frames(left, right)

    def test_duplicate_key_is_rejected(self) -> None:
        left = predictions()
        left.loc[1, ["channel_id", "prediction_time"]] = left.loc[
            0, ["channel_id", "prediction_time"]
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare_frames(left, predictions())


if __name__ == "__main__":
    unittest.main()
