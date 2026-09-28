"""Past-only rearming, conservative conflicts, month/restart and label isolation."""

from copy import deepcopy
from datetime import datetime, timedelta
import json
import unittest

from analysis.q2_recovery_reset_a import RecoveryResetPolicy
from stage1.shadow.checkpoint import CheckpointError, digest
from stage1.state_labeling.registered_episodes import StateEvent


class RecoveryResetTests(unittest.TestCase):
    start = datetime(2025, 1, 31, 20)

    def group(self, policy, hours, *values, sensor_type="Датчик дыма", channel="a"):
        at = self.start + timedelta(hours=hours)
        policy.observe_group(
            [StateEvent(i + 1, channel, sensor_type, at, value) for i, value in enumerate(values)]
        )

    def row(self, hours, normal_hours, *, eligible=True, above=True, sensor_type="Датчик дыма"):
        at = self.start + timedelta(hours=hours)
        return {
            "channel_id": "a",
            "sensor_type": sensor_type,
            "admission_status": "eligible" if eligible else "unknown",
            "admission_evidence_through": at,
            "last_explicit_normal_at": self.start + timedelta(hours=normal_hours),
            "blocking_qa_count_24h": 0,
            "availability_status": "unknown",
            "above_threshold": above,
        }

    def decide(self, policy, hours, normal_hours, **kwargs):
        return policy.decide(
            self.start + timedelta(hours=hours), [self.row(hours, normal_hours, **kwargs)]
        )[0]

    def warned(self, *, reset=True):
        policy = RecoveryResetPolicy(allow_reset=reset)
        self.group(policy, -1, "Норма")
        self.assertTrue(self.decide(policy, 0, -1)["warning_emitted"])
        return policy

    def test_reset_only_after_observed_onset_normal_and_eligible_hour(self):
        policy = self.warned()
        self.group(policy, 1, "Неисправен")
        self.assertFalse(self.decide(policy, 2, -1, eligible=False)["warning_emitted"])
        self.group(policy, 3, "Норма")
        row = self.decide(policy, 4, 3)
        self.assertEqual(row["reason"], "recovered_episode_reset")
        self.assertLess(row["previous_warning_at"], row["observed_onset_at"])
        self.assertLess(row["observed_onset_at"], row["observed_recovery_at"])
        self.assertLessEqual(row["observed_recovery_at"], row["prediction_time"])
        self.assertFalse(self.decide(policy, 5, 3)["warning_emitted"])

    def test_baseline_remains_24_hours_despite_recovery(self):
        policy = self.warned(reset=False)
        self.group(policy, 1, "Неисправен")
        self.group(policy, 2, "Норма")
        self.assertFalse(self.decide(policy, 3, 2)["warning_emitted"])
        self.assertEqual(self.decide(policy, 24, 2)["reason"], "standard_24h_warning")

    def test_false_warning_normal_only_and_unknown_history_do_not_reset(self):
        policy = self.warned()
        self.group(policy, 1, "Норма")
        self.assertFalse(self.decide(policy, 2, 1)["warning_emitted"])
        self.group(policy, 3, "Неопределен")
        self.group(policy, 4, "Неисправен")
        self.group(policy, 5, "Норма")
        self.assertFalse(self.decide(policy, 6, 5)["warning_emitted"])

    def test_conflict_same_second_and_uncertain_recovery_fail_closed(self):
        for groups in (
            [(1, ("Неисправен", "Норма")), (2, ("Норма",))],
            [(1, ("Неисправен",)), (2, ("Неопределен",)), (3, ("Норма",))],
        ):
            policy = self.warned()
            for hours, values in groups:
                self.group(policy, hours, *values)
            self.assertFalse(self.decide(policy, 4, groups[-1][0])["warning_emitted"])

    def test_type_change_does_not_rearm(self):
        policy = self.warned()
        self.group(policy, 1, "Неисправен", sensor_type="Газовый датчик")
        self.group(policy, 2, "Норма", sensor_type="Газовый датчик")
        self.assertFalse(self.decide(policy, 3, 2, sensor_type="Газовый датчик")["warning_emitted"])

    def test_qa_or_no_admission_preserves_suppression_and_token_is_not_consumed(self):
        policy = self.warned()
        self.group(policy, 1, "Неисправен")
        self.group(policy, 2, "Норма")
        self.assertFalse(self.decide(policy, 3, 2, eligible=False)["warning_emitted"])
        self.assertFalse(self.decide(policy, 4, 2, above=False)["warning_emitted"])
        self.assertTrue(self.decide(policy, 5, 2)["warning_emitted"])

    def test_later_uncertainty_invalidates_old_recovery_token(self):
        policy = self.warned()
        self.group(policy, 1, "Неисправен")
        self.group(policy, 2, "Норма")
        self.group(policy, 3, "Неопределен")
        self.group(policy, 4, "Норма")
        self.assertFalse(self.decide(policy, 5, 4)["warning_emitted"])

    def test_checkpoint_during_active_episode_cross_month_equals_continuous(self):
        original = self.warned()
        self.group(original, 2, "Неисправен")
        self.decide(original, 3, -1, eligible=False)
        restored = RecoveryResetPolicy.restore(json.loads(json.dumps(original.checkpoint())))
        for policy in (original, restored):
            self.group(policy, 4, "Норма")  # February, warning was January.
        self.assertEqual(self.decide(original, 5, 4), self.decide(restored, 5, 4))
        self.assertEqual(original.checkpoint(), restored.checkpoint())

    def test_future_append_does_not_change_prefix_or_decision(self):
        policy = self.warned()
        before = deepcopy(policy.checkpoint())
        restored = RecoveryResetPolicy.restore(before)
        self.assertEqual(self.decide(policy, 1, -1), self.decide(restored, 1, -1))
        committed = deepcopy(policy.checkpoint())
        self.group(policy, 2, "Неисправен")
        self.group(policy, 3, "Норма")
        self.assertEqual(restored.checkpoint(), committed)
        with self.assertRaises(ValueError):
            policy.decide(self.start + timedelta(hours=2), [self.row(2, -1)])

    def test_labels_future_admission_late_and_split_groups_rejected(self):
        policy = self.warned()
        row = self.row(1, -1)
        row["target"] = 1
        with self.assertRaises(ValueError):
            policy.decide(self.start + timedelta(hours=1), [row])
        row = self.row(1, 2)
        with self.assertRaises(ValueError):
            policy.decide(self.start + timedelta(hours=1), [row])
        with self.assertRaises(ValueError):
            self.group(policy, 0, "Норма")
        self.group(policy, 1, "Норма")
        with self.assertRaises(ValueError):
            self.group(policy, 1, "Неисправен")

    def test_tampered_and_future_checkpoint_rejected(self):
        policy = self.warned()
        changed = deepcopy(policy.checkpoint())
        changed["payload"]["version"] = "other"
        with self.assertRaises(CheckpointError):
            RecoveryResetPolicy.restore(changed)
        changed = deepcopy(policy.checkpoint())
        changed["payload"]["warning_states"]["a"]["last_warning_at"] = {
            "$datetime": "2025-02-02T00:00:00"
        }
        changed["sha256"] = digest(changed["payload"])
        with self.assertRaises(CheckpointError):
            RecoveryResetPolicy.restore(changed)

    def test_archive_gap_clears_state_instead_of_carrying_old_episode(self):
        policy = RecoveryResetPolicy(allow_reset=True)
        policy.observe_group([StateEvent(1, "a", "Датчик дыма", datetime(2020, 12, 31), "Норма")])
        row = self.row(0, -1)
        row.update(
            admission_evidence_through=datetime(2020, 12, 31),
            last_explicit_normal_at=datetime(2020, 12, 31),
        )
        policy.decide(datetime(2020, 12, 31, 1), [row])
        with self.assertRaises(ValueError):
            policy.observe_group([StateEvent(3, "a", "Датчик дыма", datetime(2021, 1, 1), "Норма")])
        policy.observe_group(
            [StateEvent(2, "a", "Датчик дыма", datetime(2022, 1, 1), "Неисправен")]
        )
        self.assertEqual(policy.builder.states["a"].open_episode.onset_status, "left_censored")
        self.assertIsNone(policy.warnings["a"].last_warning_at)


if __name__ == "__main__":
    unittest.main()
