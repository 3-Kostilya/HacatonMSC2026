"""R2 A checks for causal message history and separate data branches."""

from datetime import datetime, timedelta
import unittest

import pyarrow as pa

from stage1.features.hourly import FeatureEvent
from stage1.features.r2 import CompletedEpisode, build_state_history_rows, validate_r2_table


T = datetime(2025, 6, 10, 12)


def _a2_row() -> dict:
    return {
        "run_id": "a2-fixture",
        "channel_id": "smoke-1",
        "prediction_time": T,
        "sensor_type": "Датчик дыма",
        "availability_status": "unknown",
        "availability_reasons": ["cadence_unknown"],
        "baseline_status": "eligible",
        "baseline_state_count": 20,
        "baseline_numeric_count": 0,
        "baseline_numeric_median": None,
        "baseline_numeric_mad": None,
        "state_count_24h": 3,
        "numeric_count_24h": 0,
        "numeric_median_24h": None,
        "state_transitions_24h": 2,
        "excluded_quality_count_24h": 0,
        "window_reasons_24h": ["cadence_unknown"],
    }


def _event(minutes: int, state: str, alarm: bool = False, *, excluded: bool = False):
    return FeatureEvent(
        channel_id="smoke-1",
        timestamp=T + timedelta(minutes=minutes),
        alarm=alarm,
        value_state=state,
        sensor_type="Датчик дыма",
        quality_flags=("channel_time_conflict",) if excluded else (),
    )


class R2FeatureTests(unittest.TestCase):
    def test_saved_table_with_pre_r1_rules_is_rejected(self) -> None:
        table = build_state_history_rows([_a2_row()], [], source_a2_manifest_sha256="a" * 64)
        index = table.schema.get_field_index("ruleset_version")
        stale = table.set_column(
            index, table.schema.field(index), pa.array(["registered-state-r1-b1-proposal-v1"])
        )
        with self.assertRaisesRegex(ValueError, "unexpected version"):
            validate_r2_table(stale)

    def test_past_categories_and_discrete_branch_ignore_missing_numeric(self) -> None:
        events = [
            _event(-120, "Норма"),
            _event(-40, "Обнаружен дым", True),
            _event(-30, "Неисправен", False),
            _event(-10, "Неисправен", True, excluded=True),
            _event(1, "Неисправен", True),
        ]
        row = build_state_history_rows(
            [_a2_row()], events, source_a2_manifest_sha256="a" * 64
        ).to_pylist()[0]
        self.assertEqual(row["registered_fault_text_count_1h"], 1)
        self.assertEqual(row["technical_message_count_1h"], 1)
        self.assertEqual(row["environmental_alarm_count_1h"], 1)
        self.assertEqual(row["normal_message_count_1h"], 0)
        self.assertEqual(row["normal_message_count_6h"], 1)
        self.assertEqual(row["numeric_data_status"], "unknown")
        self.assertEqual(row["discrete_data_status"], "eligible")
        self.assertEqual(row["model_admission_status"], "unknown")
        self.assertEqual(row["future_label_status"], "unknown")
        self.assertIsNone(row["last_completed_episode_end_age_seconds"])

    def test_future_events_and_unfinished_episodes_do_not_change_past(self) -> None:
        past = [_event(-30, "Неисправен")]
        completed = CompletedEpisode("smoke-1", T - timedelta(hours=25), T - timedelta(hours=24))
        future_close = CompletedEpisode("smoke-1", T - timedelta(hours=1), T + timedelta(hours=1))
        first = build_state_history_rows(
            [_a2_row()],
            past,
            source_a2_manifest_sha256="a" * 64,
            completed_episodes=[completed],
        ).to_pylist()[0]
        second = build_state_history_rows(
            [_a2_row()],
            [*past, _event(5, "Неисправен")],
            source_a2_manifest_sha256="a" * 64,
            completed_episodes=[completed, future_close],
        ).to_pylist()[0]
        self.assertEqual(first, second)
        self.assertEqual(first["last_completed_episode_end_age_seconds"], 24 * 3600)
        self.assertEqual(first["completed_episode_count_168h"], 1)
        self.assertEqual(first["completed_episode_mean_duration_seconds_168h"], 3600)
        self.assertEqual(first["episode_history_status"], "unambiguous_completed_only")

    def test_completed_episode_age_does_not_cross_missing_2021(self) -> None:
        row = _a2_row()
        row["prediction_time"] = datetime(2022, 1, 2)
        old = CompletedEpisode("smoke-1", datetime(2020, 12, 30), datetime(2020, 12, 31))
        result = build_state_history_rows(
            [row],
            [],
            source_a2_manifest_sha256="a" * 64,
            completed_episodes=[old],
        ).to_pylist()[0]
        self.assertIsNone(result["last_completed_episode_end_age_seconds"])
        self.assertEqual(result["completed_episode_count_168h"], 0)

    def test_cross_archive_episode_is_rejected(self) -> None:
        invalid = CompletedEpisode("smoke-1", datetime(2020, 12, 31), datetime(2022, 1, 1))
        with self.assertRaisesRegex(ValueError, "archive boundary"):
            build_state_history_rows(
                [_a2_row()],
                [],
                source_a2_manifest_sha256="a" * 64,
                completed_episodes=[invalid],
            )

    def test_insufficient_history_does_not_become_eligible(self) -> None:
        row = _a2_row()
        row["availability_reasons"] = ["cadence_unknown", "insufficient_history"]
        result = build_state_history_rows(
            [row], [], source_a2_manifest_sha256="a" * 64
        ).to_pylist()[0]
        self.assertEqual(result["discrete_data_status"], "unknown")
        self.assertIn("insufficient_history", result["discrete_data_reasons"])


if __name__ == "__main__":
    unittest.main()
