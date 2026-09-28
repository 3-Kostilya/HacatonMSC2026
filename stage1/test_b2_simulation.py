from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from stage1.b2_validation import validate_b2_suites
from stage1.simulation import (
    B2_SCENARIO_NAMES,
    CONFIG_PATH,
    build_b2_suite,
    load_simulation_config,
    write_suite,
)


class B2SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuning = build_b2_suite("tuning")
        cls.holdout = build_b2_suite("synthetic_holdout")

    def test_expanded_suites_are_deterministic_and_independent(self):
        first = self.holdout.manifest()
        second = build_b2_suite("synthetic_holdout").manifest()
        self.assertEqual(first["events_sha256"], second["events_sha256"])
        report = validate_b2_suites(self.tuning, self.holdout)
        self.assertTrue(report["passed"], report)
        self.assertTrue(all(report["cross_suite_checks"].values()))

    def test_all_declared_b2_scenarios_and_20_channels_are_present(self):
        self.assertEqual(
            tuple(item.scenario_name for item in self.holdout.truth), B2_SCENARIO_NAMES
        )
        manifest = self.holdout.manifest()
        self.assertEqual(len(manifest["validation_channels"]), 20)
        self.assertEqual(manifest["excluded_source_years"], [2021])
        self.assertFalse(any(event.timestamp.year == 2021 for event in self.holdout.events))

    def test_stuck_differs_from_dropout_and_neither_invents_zero(self):
        truth = {item.scenario_name: item for item in self.holdout.truth}
        stuck = truth["numeric_stuck"]
        stuck_events = [
            event
            for event in self.holdout.events
            if event.source == f"synthetic:{stuck.scenario_id}"
            and stuck.intervention_start <= event.timestamp < stuck.end
        ]
        dropout = truth["known_cadence_dropout"]
        dropout_events = [
            event
            for event in self.holdout.events
            if event.source == f"synthetic:{dropout.scenario_id}"
        ]
        self.assertGreater(len(stuck_events), 1)
        self.assertEqual(len({event.numeric_value for event in stuck_events}), 1)
        self.assertFalse(
            any(
                dropout.intervention_start <= event.timestamp < dropout.end
                for event in dropout_events
            )
        )
        self.assertFalse(any(event.raw_value == "0" for event in dropout_events))

    def test_mixed_channel_keeps_numeric_and_state_observations(self):
        truth = next(
            item for item in self.holdout.truth if item.scenario_name == "mixed_numeric_state"
        )
        events = [
            event
            for event in self.holdout.events
            if event.source == f"synthetic:{truth.scenario_id}"
        ]
        self.assertEqual({event.channel_id for event in events}, set(truth.channel_ids))
        self.assertTrue(any(event.numeric_value is not None for event in events))
        self.assertTrue(any(event.numeric_value is None for event in events))

    def test_export_gap_and_common_environment_are_controls(self):
        truth = {item.scenario_name: item for item in self.holdout.truth}
        export_gap = truth["export_gap_control"]
        environment = truth["common_environment_control"]
        self.assertEqual((export_gap.label, environment.label), ("control", "control"))
        self.assertIsNone(export_gap.failure_point)
        self.assertIsNone(environment.failure_point)
        for scenario in (export_gap, environment):
            self.assertGreaterEqual(len(scenario.channel_ids), 3)

    def test_manifest_hashes_match_written_events(self):
        with tempfile.TemporaryDirectory() as directory:
            events_path, manifest_path = write_suite(self.holdout, Path(directory))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            import hashlib

            self.assertEqual(
                hashlib.sha256(events_path.read_bytes()).hexdigest(), manifest["events_sha256"]
            )
            self.assertTrue(all(len(item["event_sha256"]) == 64 for item in manifest["scenarios"]))

    def test_excluded_2021_segment_is_rejected(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["suites"]["tuning"]["segment_start"] = "2021-03-01 00:00:00"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "excluded source year"):
                load_simulation_config(path)


if __name__ == "__main__":
    unittest.main()
