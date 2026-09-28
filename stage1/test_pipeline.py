from datetime import datetime, timedelta
import unittest

from stage1.contracts import Decision, NormalizedEvent
from stage1.pipeline import evaluate_catalog, evaluate_channel


def numeric_event(channel: str, hour: int, value: float) -> NormalizedEvent:
    return NormalizedEvent(
        channel_id=channel,
        timestamp=datetime(2026, 1, 1) + timedelta(hours=hour),
        raw_value=str(value),
        numeric_value=value,
        alarm=False,
        sensor_type="Датчик температуры",
        source="fixture.csv",
    )


class PipelineTests(unittest.TestCase):
    def test_registry_routes_numeric_type_and_keeps_context_unknown(self):
        values = [10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 11, 9, 30, 31, 32]
        result = evaluate_channel(
            [numeric_event("1", hour, value) for hour, value in enumerate(values)]
        )
        self.assertEqual(result.processing_mode, "numeric")
        self.assertEqual(result.detector_result.decision, Decision.CANDIDATE)
        self.assertEqual(result.context_status, "unknown")
        self.assertEqual(result.context_reason, "unconfirmed_context_link")
        self.assertEqual(result.observability.status, "unknown")

    def test_catalog_keeps_channels_separate(self):
        events = [numeric_event("1", hour, 10) for hour in range(15)]
        events.extend(numeric_event("2", hour, 20) for hour in range(15))
        outcomes = evaluate_catalog(events)
        self.assertEqual([outcome.channel_id for outcome in outcomes], ["1", "2"])

    def test_unknown_type_is_rejected_instead_of_guessed(self):
        unknown = NormalizedEvent(
            channel_id="x",
            timestamp=datetime(2026, 1, 1),
            raw_value="Норма",
            alarm=False,
            sensor_type="unknown",
            source="fixture.csv",
            quality_flags=("unknown_channel",),
        )
        with self.assertRaisesRegex(KeyError, "absent from the validated registry"):
            evaluate_catalog([unknown])

    def test_recovery_allows_two_sequential_numeric_episodes(self):
        values = [10] * 12 + [20] * 3 + [10] * 20 + [20] * 3 + [10] * 20
        result = evaluate_channel(
            [numeric_event("1", hour, value) for hour, value in enumerate(values)]
        )
        candidates = [
            episode for episode in result.detector_results if episode.decision is Decision.CANDIDATE
        ]
        self.assertEqual(len(candidates), 2)
        self.assertIsNotNone(candidates[0].end_at)
        self.assertLess(candidates[0].end_at, candidates[1].start_at)


if __name__ == "__main__":
    unittest.main()
