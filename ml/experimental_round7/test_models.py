"""Small tests for the pinned research bundle; no source dataset needed."""

from __future__ import annotations

import unittest

import pandas as pd

from ml.experimental_round7.models import Round7ResearchModels


class Round7ResearchModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = Round7ResearchModels()

    def row(self, *, status: str = "eligible", sensor_type: str = "Датчик дыма") -> dict:
        row = {name: 0 for name in self.models.base_names}
        row.update(
            channel_id="test-channel",
            prediction_time=pd.Timestamp("2025-01-09"),
            last_explicit_normal_at=pd.Timestamp("2025-01-08"),
            admission_status=status,
            sensor_type=sensor_type,
        )
        return row

    def test_pinned_scores_and_no_warning_emission(self) -> None:
        result = self.models.score(pd.DataFrame([self.row()])).iloc[0]
        self.assertEqual(result.prediction_status, "scored_research")
        self.assertAlmostEqual(result.score_linear, 0.9958948276297472, places=12)
        self.assertAlmostEqual(result.score_tree, 0.37889605212739164, places=12)
        self.assertAlmostEqual(result.score_specialist, 0.6114615481834429, places=12)
        self.assertFalse(result.passes_common_gates)
        self.assertFalse(result.passes_standard_gates)
        self.assertNotIn("warning", result.index)

    def test_unavailable_row_stays_unscored(self) -> None:
        result = self.models.score(pd.DataFrame([self.row(status="unknown")])).iloc[0]
        self.assertEqual(result.prediction_status, "not_available")
        self.assertTrue(pd.isna(result.score_linear))
        self.assertTrue(pd.isna(result.passes_standard_gates))

    def test_future_labels_are_rejected(self) -> None:
        row = self.row()
        row["target"] = 1
        with self.assertRaisesRegex(ValueError, "future/label"):
            self.models.score(pd.DataFrame([row]))


if __name__ == "__main__":
    unittest.main()
