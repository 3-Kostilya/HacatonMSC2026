"""A-to-B warning and cooldown semantics on an explicit past-only event trace."""

from __future__ import annotations

from datetime import datetime, timedelta
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.replay_shadow_pilot import to_b_input
from analysis.shadow_checkpoint_bundle import ReplaySession, load_bundle, save_bundle
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


def advance(session: ReplaySession, rows: list[Observation], stop: datetime) -> None:
    pending = [row for row in rows if session.a.watermark is None
               or row.event.timestamp > session.a.watermark]
    ordered = sorted(pending, key=lambda row: (row.event.timestamp, row.event.channel_id))
    groups = iter(groupby(ordered, key=lambda row: (row.event.timestamp,
                                                  row.event.channel_id)))
    current = next(groups, None)
    while session.next_prediction < stop:
        while current is not None and current[0][0] <= session.next_prediction:
            session.observe_group(list(current[1]))
            current = next(groups, None)
        session.predict_hour()


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

    def test_a_b_bundle_preserves_positive_warning_and_cooldown(self) -> None:
        policy = ShadowPolicy.from_freeze(Path("ml/r6_frozen_rule_v1.json"))
        source = {"mode": "b_positive_warning_restart_test"}
        def session() -> ReplaySession:
            return ReplaySession(source_identity=source, channels=["channel-1"],
                                 start=T, end=T + timedelta(hours=25), policy=policy)

        rows = [observation(T - timedelta(days=35 - i), row_id=i + 1)
                for i in range(26)]
        rows += [observation(T - timedelta(hours=3), row_id=40),
                 *[observation(T - timedelta(hours=2), "Неисправен", 50 + i)
                   for i in range(4)],
                 observation(T - timedelta(hours=1), row_id=60),
                 *[observation(T + timedelta(hours=22), "Неисправен", 70 + i)
                   for i in range(4)],
                 observation(T + timedelta(hours=23), row_id=80)]
        full, prefix = session(), session()
        advance(full, rows, full.end)
        advance(prefix, rows, T + timedelta(hours=1))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint"
            save_bundle(prefix, path)
            resumed = load_bundle(path, source_identity=source, policy=policy)
            advance(resumed, rows, resumed.end)
        self.assertEqual(resumed.predictions, full.predictions)
        self.assertEqual(resumed.decisions, full.decisions)
        self.assertEqual(resumed.checkpoint(), full.checkpoint())
        self.assertEqual([row["prediction_time"] for row in resumed.decisions
                          if row["shadow_warning"]],
                         [T.isoformat(), (T + timedelta(hours=24)).isoformat()])
        self.assertEqual(resumed.decisions[1]["warning_reason"], "channel_cooldown")
        self.assertTrue(all(not row["automatic_action_taken"] for row in resumed.decisions))


if __name__ == "__main__":
    unittest.main()
