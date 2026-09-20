from datetime import datetime, timedelta
import unittest

from stage1.contracts import NormalizedEvent
from stage1.observability import AuditStatus, ObservabilityPolicy, audit_channel


def event(timestamp: datetime, *flags: str) -> NormalizedEvent:
    return NormalizedEvent(
        channel_id="42",
        timestamp=timestamp,
        raw_value="1",
        numeric_value=1.0,
        alarm=False,
        sensor_type="Датчик температуры",
        source="test.csv",
        quality_flags=flags,
    )


class ObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.decision_at = datetime(2026, 1, 8, 12)
        self.policy = ObservabilityPolicy(
            expected_cadence=timedelta(minutes=30),
            minimum_history=timedelta(days=7),
            minimum_coverage=0.5,
        )

    def test_future_events_never_change_audit(self):
        past = [event(self.decision_at - timedelta(days=7))]
        past.extend(
            event(self.decision_at - timedelta(minutes=30 * offset)) for offset in range(0, 15)
        )
        baseline = audit_channel("42", past, self.decision_at, self.policy)
        with_future = audit_channel(
            "42",
            past + [event(self.decision_at + timedelta(minutes=30))],
            self.decision_at,
            self.policy,
        )
        self.assertEqual(baseline, with_future)

    def test_large_gap_is_unknown_and_is_not_filled(self):
        events = [
            event(self.decision_at - timedelta(days=7)),
            event(self.decision_at - timedelta(minutes=55)),
            event(self.decision_at - timedelta(minutes=5)),
        ]
        policy = ObservabilityPolicy(
            expected_cadence=timedelta(minutes=5),
            minimum_history=timedelta(days=7),
            minimum_coverage=0.1,
        )
        window = audit_channel("42", events, self.decision_at, policy).window("1h")
        self.assertEqual(window.status, AuditStatus.UNKNOWN)
        self.assertIn("large_gap", window.flags)
        self.assertEqual(window.event_count, 2)
        self.assertEqual(len(window.observed_intervals), 2)
        self.assertEqual(window.observed_intervals[0].start_at, events[1].timestamp)
        self.assertEqual(window.observed_intervals[0].end_at, events[1].timestamp)

    def test_unknown_cadence_keeps_coverage_unknown(self):
        report = audit_channel(
            "42",
            [event(self.decision_at - timedelta(days=8)), event(self.decision_at)],
            self.decision_at,
            ObservabilityPolicy(expected_cadence=None),
        )
        self.assertEqual(report.status, AuditStatus.UNKNOWN)
        self.assertEqual(report.window("1h").status, AuditStatus.UNKNOWN)
        self.assertIsNone(report.window("1h").coverage)
        self.assertIn("cadence_unknown", report.reasons)

    def test_insufficient_history_is_unknown_not_excluded(self):
        events = [event(self.decision_at - timedelta(hours=offset)) for offset in range(8)]
        report = audit_channel("42", events, self.decision_at, self.policy)
        self.assertFalse(report.history_sufficient)
        self.assertEqual(report.status, AuditStatus.UNKNOWN)
        self.assertIn("insufficient_history", report.reasons)

    def test_quality_only_interval_is_explicitly_excluded(self):
        policy = ObservabilityPolicy(
            expected_cadence=timedelta(minutes=30),
            excluded_quality_flags=frozenset({"channel_time_conflict"}),
        )
        report = audit_channel(
            "42",
            [event(self.decision_at, "channel_time_conflict")],
            self.decision_at,
            policy,
        )
        self.assertEqual(report.status, AuditStatus.EXCLUDE)
        self.assertEqual(report.window("1h").status, AuditStatus.EXCLUDE)
        self.assertEqual(report.window("1h").reasons, ("all_interval_observations_excluded",))

    def test_complete_causal_windows_are_included(self):
        events = [event(self.decision_at - timedelta(days=7))]
        events.extend(
            event(self.decision_at - timedelta(minutes=30 * offset))
            for offset in range(0, 7 * 24 * 2 + 1)
        )
        report = audit_channel("42", events, self.decision_at, self.policy)
        self.assertEqual(report.status, AuditStatus.INCLUDE)
        self.assertTrue(report.history_sufficient)
        self.assertEqual([item.name for item in report.windows], ["1h", "6h", "24h", "7d"])
        self.assertTrue(all(item.status is AuditStatus.INCLUDE for item in report.windows))

    def test_explicit_channel_exclusion_applies_to_each_interval(self):
        report = audit_channel(
            "42",
            [event(self.decision_at)],
            self.decision_at,
            self.policy,
            exclusion_reasons=("channel_mapping_invalid",),
        )
        self.assertEqual(report.status, AuditStatus.EXCLUDE)
        self.assertTrue(all(item.status is AuditStatus.EXCLUDE for item in report.windows))
        self.assertEqual(report.window("1h").reasons, ("channel_mapping_invalid",))


if __name__ == "__main__":
    unittest.main()
