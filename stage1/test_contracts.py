from datetime import datetime, timezone
import unittest

from stage1.contracts import Decision, Episode, NormalizedEvent


class NormalizedEventTests(unittest.TestCase):
    def test_preserves_text_and_numeric_views(self):
        event = NormalizedEvent(
            channel_id="42",
            timestamp=datetime(2026, 1, 2, 3, 4, 5),
            raw_value="12.5",
            numeric_value=12.5,
            alarm=False,
            sensor_type="Датчик температуры",
            source="ext-journal-2026.7z",
        )
        self.assertEqual(event.to_record()["raw_value"], "12.5")
        self.assertEqual(event.to_record()["numeric_value"], 12.5)

    def test_rejects_timezone_conversion(self):
        with self.assertRaisesRegex(ValueError, "local naive time"):
            NormalizedEvent(
                channel_id="42",
                timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
                raw_value="Норма",
                alarm=False,
                sensor_type="Датчик дыма",
                source="sample.csv",
            )


class EpisodeTests(unittest.TestCase):
    def make_episode(self, **overrides):
        values = {
            "episode_id": "ep-1",
            "channel_id": "42",
            "sensor_type": "Датчик дыма",
            "sensor_group": "fire_discrete",
            "anomaly_type": "rapid_switching",
            "decision": Decision.CANDIDATE,
            "start_at": datetime(2026, 1, 2, 3),
            "confirmed_at": datetime(2026, 1, 2, 4),
            "ruleset_version": "stage1-v1",
            "evidence": ("switch_count_1h=9",),
            "observation_quality": (),
        }
        values.update(overrides)
        return Episode(**values)

    def test_candidate_requires_evidence(self):
        with self.assertRaisesRegex(ValueError, "require at least one evidence"):
            self.make_episode(evidence=())

    def test_unknown_requires_reason(self):
        with self.assertRaisesRegex(ValueError, "observation-quality reason"):
            self.make_episode(decision=Decision.UNKNOWN, evidence=(), observation_quality=())

    def test_rejects_confirmation_before_start(self):
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            self.make_episode(confirmed_at=datetime(2026, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()
