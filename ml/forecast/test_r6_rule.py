"""The final rule reproduces R4 while preserving unavailable predictions."""

import unittest

import pandas as pd

from analysis.train_r4_discrete_baselines import rule_score
from ml.forecast.r6_rule import predict_rule


class R6RuleTest(unittest.TestCase):
    def test_eligible_scores_match_accepted_r4_and_unknown_is_unavailable(self) -> None:
        frame = pd.DataFrame({
            "registered_fault_text_count_24h": [3, None, 10],
            "registered_fault_text_count_168h": [2, 0, 10],
            "completed_episode_count_168h": [1, 0, 10],
            "technical_message_count_24h": [0, 0, 10],
        })
        status = pd.Series(["eligible", "eligible", "unknown"])
        scored = predict_rule(frame, eligibility_status=status, threshold=7.1)
        self.assertEqual(scored.rule_score.iloc[:2].tolist(),
                         rule_score(frame.iloc[:2]).tolist())
        self.assertEqual(scored.alert.iloc[:2].tolist(), [True, False])
        self.assertTrue(pd.isna(scored.rule_score.iloc[2]))
        self.assertTrue(pd.isna(scored.alert.iloc[2]))

    def test_missing_admission_is_rejected(self) -> None:
        frame = pd.DataFrame({name: [0] for name in (
            "registered_fault_text_count_24h",
            "registered_fault_text_count_168h",
            "completed_episode_count_168h",
            "technical_message_count_24h",
        )})
        with self.assertRaisesRegex(ValueError, "explicit"):
            predict_rule(frame, eligibility_status=pd.Series([None]), threshold=7.1)


if __name__ == "__main__":
    unittest.main()
