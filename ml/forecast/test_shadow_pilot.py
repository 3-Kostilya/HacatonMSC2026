"""Causal boundaries and operator-facing semantics of the shadow policy."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import unittest

import pandas as pd

from ml.forecast.r6_rule import predict_rule
from ml.forecast.shadow_pilot import (
    ShadowPolicy, ShadowState, decide_shadow_hour, run_shadow_batch,
    summarize_shadow_batch,
)


FREEZE = Path("ml/r6_frozen_rule_v1.json")


def row(at: datetime, **overrides: object) -> dict:
    value = {
        "channel_id": "smoke-1",
        "sensor_type": "Датчик дыма",
        "prediction_time": at,
        "admission_status": "eligible",
        "admission_reason": None,
        "history_through": at,
        "admission_through": at,
        "registered_fault_text_count_24h": 3,
        "registered_fault_text_count_168h": 2,
        "completed_episode_count_168h": 0,
        "technical_message_count_24h": 1,
    }
    value.update(overrides)
    return value


class ShadowPilotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = ShadowPolicy.from_freeze(FREEZE)

    def test_frozen_score_cooldown_and_checkpoint_across_batches(self) -> None:
        at = datetime(2026, 6, 30, 23)
        state = ShadowState()
        first = decide_shadow_hour(row(at), state, self.policy)
        accepted = predict_rule(pd.DataFrame([row(at)]),
                                eligibility_status="eligible", threshold=7.1)
        self.assertEqual(first["rule_score"], accepted.rule_score.iloc[0])
        self.assertEqual(sum(first["score_contributions"].values()), 7.1)
        self.assertTrue(first["shadow_warning"])
        self.assertEqual(first["delivery_mode"], "record_only")
        self.assertFalse(first["automatic_action_taken"])

        resumed = ShadowState.restore(state.checkpoint(self.policy), self.policy)
        decisions = run_shadow_batch([row(at + timedelta(hours=23)),
                                      row(at + timedelta(hours=24))],
                                     resumed, self.policy)
        self.assertEqual([item["shadow_warning"] for item in decisions],
                         [False, True])
        self.assertEqual(decisions[0]["warning_reason"], "channel_cooldown")

    def test_unknown_and_invalid_counts_remain_unavailable(self) -> None:
        at = datetime(2026, 6, 1)
        state = ShadowState()
        unknown = decide_shadow_hour(row(at, admission_status="unknown",
                                         admission_reason="coverage_unverified"),
                                     state, self.policy)
        invalid = decide_shadow_hour(row(at + timedelta(hours=1),
                                         registered_fault_text_count_24h=None),
                                     state, self.policy)
        self.assertEqual(unknown["unavailable_reason"], "coverage_unverified")
        self.assertEqual(invalid["unavailable_reason"],
                         "missing_or_invalid_rule_count")
        self.assertIsNone(unknown["rule_score"])
        self.assertIsNone(invalid["threshold_crossed"])
        self.assertFalse(unknown["shadow_warning"])

    def test_future_inputs_and_duplicate_hour_are_rejected_or_withheld(self) -> None:
        at = datetime(2026, 6, 1)
        state = ShadowState()
        with self.assertRaisesRegex(ValueError, "future-label"):
            decide_shadow_hour(row(at, target=1), state, self.policy)
        with self.assertRaisesRegex(ValueError, "future-label"):
            decide_shadow_hour(row(at, label_status="positive"), state, self.policy)
        self.assertEqual(state.last_prediction_at, {})
        future = decide_shadow_hour(row(at, admission_through=at + timedelta(hours=1)),
                                    state, self.policy)
        self.assertEqual(future["unavailable_reason"], "future_evidence")
        with self.assertRaisesRegex(ValueError, "chronological"):
            decide_shadow_hour(row(at), state, self.policy)

    def test_missing_admission_reason_and_wrong_freeze_are_rejected(self) -> None:
        at = datetime(2026, 6, 1)
        with self.assertRaisesRegex(ValueError, "reason"):
            decide_shadow_hour(row(at, admission_status="excluded",
                                   admission_reason=None), ShadowState(), self.policy)
        checkpoint = ShadowState().checkpoint(self.policy)
        checkpoint["freeze_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "another policy"):
            ShadowState.restore(checkpoint, self.policy)

    def test_summary_counts_only_observable_load(self) -> None:
        at = datetime(2026, 6, 1)
        state = ShadowState()
        decisions = run_shadow_batch([
            row(at),
            row(at + timedelta(hours=1), admission_status="unknown",
                admission_reason="coverage_unverified"),
        ], state, self.policy)
        report = summarize_shadow_batch(decisions, expected_channel_hours=3)
        self.assertEqual(report["scored_hours"], 1)
        self.assertEqual(report["unavailable_reasons"], {"coverage_unverified": 1})
        self.assertEqual(report["coverage_of_expected_hours"], 1 / 3)
        self.assertEqual(report["shadow_warnings"], 1)
        self.assertFalse(report["future_label_metrics_available"])


if __name__ == "__main__":
    unittest.main()
