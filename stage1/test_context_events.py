from datetime import datetime, timedelta
import unittest

from stage1.context_events import detect_shared_state
from stage1.contracts import Decision, NormalizedEvent, Origin


def event(channel: str, minute: int, value: str, object_id: str | None) -> NormalizedEvent:
    return NormalizedEvent(
        channel_id=channel,
        timestamp=datetime(2026, 1, 1) + timedelta(minutes=minute),
        raw_value=value,
        alarm=False,
        sensor_type="Состояние фазы",
        source="synthetic:outage",
        object_id=object_id,
    )


class SharedContextTests(unittest.TestCase):
    def test_emits_one_shared_episode_for_multiple_channels(self):
        result = detect_shared_state(
            [event("a", 10, "Нет связи", "obj"), event("b", 11, "Нет связи", "obj")],
            state="Нет связи",
        )
        self.assertEqual(result.decision, Decision.CANDIDATE)
        self.assertEqual(result.channel_id, "context:obj")
        self.assertEqual(result.origin, Origin.SYNTHETIC)
        self.assertIn("distinct_channels=2", result.evidence)

    def test_missing_link_is_unknown(self):
        result = detect_shared_state(
            [event("a", 10, "Нет связи", None), event("b", 11, "Нет связи", None)],
            state="Нет связи",
        )
        self.assertEqual(result.decision, Decision.UNKNOWN)
        self.assertIn("unconfirmed_context_link", result.observation_quality)

    def test_unrelated_times_do_not_form_shared_episode(self):
        result = detect_shared_state(
            [event("a", 0, "Нет связи", "obj"), event("b", 60, "Нет связи", "obj")],
            state="Нет связи",
        )
        self.assertEqual(result.decision, Decision.NO_CANDIDATE)


if __name__ == "__main__":
    unittest.main()
