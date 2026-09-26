"""A-to-B warning and cooldown semantics on an explicit past-only event trace."""

from __future__ import annotations

from datetime import datetime, timedelta
from itertools import groupby
from pathlib import Path
import unittest

from analysis.replay_shadow_pilot import to_b_input
from ml.forecast.shadow_pilot import ShadowPolicy, ShadowState, decide_shadow_hour
from stage1.features.hourly import FeatureEvent
from stage1.shadow.stream import Observation, ShadowStream


T = datetime(2025, 12, 1)


def observation(at: datetime, text: str = "Норма", row_id: int = 1) -> Observation:
    return Observation(
        row_id,
        FeatureEvent("channel-1", at, False, value_state=text,
                     sensor_type="Датчик дыма"),
        f"ext-journal-{at.year}.7z",
    )


def feed(stream: ShadowStream, rows: list[Observation]) -> None:
    ordered = sorted(rows, key=lambda row: (row.event.timestamp, row.event.channel_id))
    for _, group in groupby(ordered, key=lambda row: (row.event.timestamp,
                                                     row.event.channel_id)):
        stream.observe_group(list(group))


class ShadowWarningPathTest(unittest.TestCase):
    def test_unchanged_warning_rule_survives_b_checkpoint_and_cooldown(self) -> None:
        policy = ShadowPolicy.from_freeze(Path("ml/r6_frozen_rule_v1.json"))
        stream = ShadowStream(threshold=7.1)
        state = ShadowState()
        baseline = [observation(T - timedelta(days=35 - i), row_id=i + 1)
                    for i in range(26)]
        feed(stream, [*baseline, observation(T - timedelta(hours=3), row_id=40),
                      *[observation(T - timedelta(hours=2), "Неисправен", 50 + i)
                        for i in range(4)],
                      observation(T - timedelta(hours=1), row_id=60)])

        first_a = stream.predict(T, ["channel-1"])[0]
        first_b = decide_shadow_hour(to_b_input(first_a), state, policy)
        self.assertEqual(first_a["admission_status"], "eligible")
        self.assertTrue(first_a["warning_emitted"])
        self.assertTrue(first_b["shadow_warning"])
        self.assertEqual(first_a["rule_score"], first_b["rule_score"])
        self.assertFalse(first_b["automatic_action_taken"])

        resumed = ShadowState.restore(state.checkpoint(policy), policy)
        second_a = stream.predict(T + timedelta(hours=1), ["channel-1"])[0]
        second_b = decide_shadow_hour(to_b_input(second_a), resumed, policy)
        self.assertEqual(second_a["warning_status"], "suppressed_cooldown")
        self.assertEqual(second_b["warning_reason"], "channel_cooldown")
        self.assertFalse(second_b["shadow_warning"])

        feed(stream, [*[observation(T + timedelta(hours=22), "Неисправен", 70 + i)
                       for i in range(4)],
                      observation(T + timedelta(hours=23), row_id=80)])
        third_a = stream.predict(T + timedelta(hours=24), ["channel-1"])[0]
        third_b = decide_shadow_hour(to_b_input(third_a), resumed, policy)
        self.assertTrue(third_a["warning_emitted"])
        self.assertTrue(third_b["shadow_warning"])
        self.assertEqual(third_a["rule_score"], third_b["rule_score"])
        self.assertEqual(third_b["delivery_mode"], "record_only")


if __name__ == "__main__":
    unittest.main()
