"""The first R3 discrete baseline has a fixed, past-only feature contract."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import unittest

from analysis.build_r3_full_month import FULL_PACK_VERSION
from stage1.features.hourly import FeatureEvent
from stage1.features.r2 import CATEGORY_FIELDS, CompletedEpisode, build_state_history_rows
from stage1.features.r3 import FEATURE_PACK_SCHEMA, MODEL_FEATURE_ALLOWLIST, ROW_STATUS_SCHEMA
from stage1.features.r3_full import build_selected_hourly_rows
from stage1.features.schema import WINDOW_HOURS


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "ml" / "r3_discrete_feature_allowlist_v1.json"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


class R3DiscreteAllowlistTests(unittest.TestCase):
    def test_exact_feature_groups_are_subset_of_a3_and_exclude_leakage(self) -> None:
        contract = _contract()
        expected = (
            "sensor_type",
            "last_observation_age_seconds",
            "baseline_event_count",
            "baseline_state_count",
            *(
                f"{name}_{hours}h"
                for hours in WINDOW_HOURS
                for name in (
                    "event_count", "alarm_count", "state_count", "state_transitions",
                    "state_distinct_count", "maximum_gap_seconds",
                )
            ),
            *(f"{name}_{hours}h" for hours in WINDOW_HOURS for name in CATEGORY_FIELDS),
            "last_completed_episode_end_age_seconds",
            "completed_episode_count_168h",
            "completed_episode_mean_duration_seconds_168h",
        )
        names = tuple(contract["feature_names"])
        self.assertEqual(contract["source_feature_pack_version"], FULL_PACK_VERSION)
        self.assertEqual(contract["admission_rule_version"], "r3-b-conditional-discrete-v1")
        self.assertEqual(names, expected)
        self.assertEqual(len(names), contract["feature_count"])
        self.assertEqual(len(names), 51)
        for name in ("source_a3_manifest_sha256", "source_b3_manifest_sha256"):
            self.assertEqual(len(contract[name]), 64)
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(set(names) <= set(MODEL_FEATURE_ALLOWLIST))
        self.assertTrue(set(names) <= set(FEATURE_PACK_SCHEMA.names))
        self.assertFalse(set(names) & set(ROW_STATUS_SCHEMA.names))
        self.assertEqual(contract["categorical_feature_names"], ["sensor_type"])
        self.assertFalse(any("numeric" in name or "coverage" in name for name in names))
        self.assertFalse(set(names) & {
            "channel_id", "prediction_time", "target", "target_episode_id",
            "label_available_at", "split", "admission_status", "baseline_dominant_state",
        })

    def test_future_events_and_unfinished_episodes_cannot_change_features(self) -> None:
        at = datetime(2025, 6, 10, 12)
        past = [
            FeatureEvent("c", at - timedelta(days=20), False,
                         value_state="Норма", sensor_type="Датчик дыма"),
            FeatureEvent("c", at - timedelta(hours=1), False,
                         value_state="Норма", sensor_type="Датчик дыма"),
        ]
        future = FeatureEvent("c", at + timedelta(hours=1), True,
                              value_state="Неисправен", sensor_type="Датчик дыма")
        closed = CompletedEpisode("c", at - timedelta(hours=48), at - timedelta(hours=24))
        future_closed = CompletedEpisode("c", at + timedelta(hours=1),
                                         at + timedelta(hours=2))

        def selected(events: list[FeatureEvent], episodes: list[CompletedEpisode]) -> dict:
            row = build_selected_hourly_rows(events, "c", [at])[0]
            row["run_id"] = "discrete-allowlist-test"
            history = build_state_history_rows(
                [row], events, source_a2_manifest_sha256="a" * 64,
                completed_episodes=episodes,
            ).to_pylist()[0]
            combined = {**row, **history}
            return {name: combined[name] for name in _contract()["feature_names"]}

        self.assertEqual(selected(past, [closed]), selected([*past, future], [closed, future_closed]))


if __name__ == "__main__":
    unittest.main()
