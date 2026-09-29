from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from stage1.contracts import NormalizedEvent
from stage1.detectors import (
    detect_context_coincidence,
    detect_discrete_pattern,
    detect_numeric_level_shift,
)
from stage1.simulation import SCENARIO_NAMES, build_suite, write_suite


def _signature(suite):
    return [event.to_record() for event in suite.events], suite.manifest()


class SimulationTests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        self.assertEqual(
            _signature(build_suite("synthetic_holdout")),
            _signature(build_suite("synthetic_holdout")),
        )

    def test_all_frozen_scenarios_are_present_once(self) -> None:
        suite = build_suite("synthetic_holdout")
        self.assertEqual(tuple(item.scenario_name for item in suite.truth), SCENARIO_NAMES)
        self.assertEqual(len({item.scenario_id for item in suite.truth}), len(SCENARIO_NAMES))
        self.assertTrue(all(isinstance(event, NormalizedEvent) for event in suite.events))
        types_by_channel: dict[str, set[str]] = {}
        for event in suite.events:
            types_by_channel.setdefault(event.channel_id, set()).add(event.sensor_type)
        self.assertTrue(all(len(sensor_types) == 1 for sensor_types in types_by_channel.values()))

    def test_interventions_do_not_change_clean_prefix(self) -> None:
        suite = build_suite("synthetic_holdout")
        numeric_prefixes = []
        for truth in suite.truth:
            scenario_events = [
                event
                for event in suite.events
                if event.source == f"synthetic:{truth.scenario_id}"
                and event.timestamp < truth.intervention_start
            ]
            self.assertTrue(scenario_events)
            self.assertTrue(
                all("synthetic_observation" in event.quality_flags for event in scenario_events)
            )
            self.assertGreaterEqual(
                (
                    truth.intervention_start - min(event.timestamp for event in scenario_events)
                ).total_seconds(),
                604800,
            )
            if truth.scenario_name.startswith("numeric_"):
                self.assertTrue(all(event.numeric_value is not None for event in scenario_events))
                numeric_prefixes.append(
                    [(event.timestamp, event.numeric_value) for event in scenario_events]
                )
        self.assertTrue(all(prefix == numeric_prefixes[0] for prefix in numeric_prefixes[1:]))

    def test_tuning_and_holdout_use_different_segments_and_parameters(self) -> None:
        tuning = build_suite("tuning")
        holdout = build_suite("synthetic_holdout")
        self.assertNotEqual(tuning.parameters["segment_start"], holdout.parameters["segment_start"])
        self.assertNotEqual(tuning.parameters, holdout.parameters)
        self.assertTrue(
            {item.scenario_id for item in tuning.truth}.isdisjoint(
                item.scenario_id for item in holdout.truth
            )
        )

    def test_truth_has_required_boundaries_and_expected_behavior(self) -> None:
        suite = build_suite("synthetic_holdout")
        for truth in suite.truth:
            self.assertLess(truth.intervention_start, truth.end)
            self.assertIn(truth.label, {"positive", "control"})
            self.assertTrue(truth.expected_detector_behavior)
            self.assertTrue(truth.detector_applicability)
            if truth.label == "positive":
                self.assertIsNotNone(truth.failure_point)
            else:
                self.assertIsNone(truth.failure_point)

    def test_unknown_cadence_gap_is_not_encoded_as_failure(self) -> None:
        suite = build_suite("synthetic_holdout")
        truth = next(
            item for item in suite.truth if item.scenario_name == "missingness_unknown_control"
        )
        events = [event for event in suite.events if event.channel_id in truth.channel_ids]
        self.assertIsNone(truth.expected_cadence_seconds)
        self.assertEqual(truth.expected_detector_behavior, "unknown")
        self.assertIn("cadence_unknown", truth.detector_applicability)
        self.assertFalse(
            any(truth.intervention_start <= event.timestamp <= truth.end for event in events)
        )
        self.assertFalse(any("неисправ" in event.raw_value.casefold() for event in events))

    def test_common_outage_is_one_context_truth_for_many_channels(self) -> None:
        suite = build_suite("synthetic_holdout")
        truth = next(item for item in suite.truth if item.scenario_name == "common_outage_context")
        self.assertEqual(len(truth.channel_ids), suite.parameters["outage_channel_count"])
        self.assertEqual(truth.cause_hypothesis, "common_outage")
        self.assertEqual(truth.expected_detector_behavior, "single_common_context_candidate")
        events = [event for event in suite.events if event.channel_id in truth.channel_ids]
        object_ids = {event.object_id for event in events}
        self.assertEqual(len(object_ids), 1)
        self.assertNotIn(None, object_ids)
        self.assertIn("explicit synthetic relation", " ".join(truth.notes))
        target = [event for event in events if event.channel_id == truth.channel_ids[0]]
        context = [event for event in events if event.channel_id != truth.channel_ids[0]]
        result = detect_context_coincidence(target, context)
        self.assertEqual(result.decision.value, "candidate")
        self.assertEqual(result.start_at, truth.intervention_start)

    def test_baseline_detectors_obey_positive_and_control_cases(self) -> None:
        suite = build_suite("synthetic_holdout")
        for truth in suite.truth:
            if truth.detector_applicability not in {"numeric", "discrete"}:
                continue
            events = [event for event in suite.events if event.channel_id in truth.channel_ids]
            result = (
                detect_numeric_level_shift(events)
                if truth.detector_applicability == "numeric"
                else detect_discrete_pattern(events)
            )
            expected = "candidate" if truth.label == "positive" else "no_candidate"
            self.assertEqual(result.decision.value, expected, truth.scenario_name)
            if truth.label == "positive":
                self.assertGreaterEqual(result.start_at, truth.intervention_start)

    def test_write_suite_emits_jsonl_events_and_truth_manifest(self) -> None:
        suite = build_suite("synthetic_holdout")
        with tempfile.TemporaryDirectory() as directory:
            events_path, manifest_path = write_suite(suite, Path(directory))
            first_event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(first_event["source"].startswith("synthetic:"))
        self.assertEqual(manifest["seed"], 260919)
        self.assertEqual(manifest["event_count"], len(suite.events))


if __name__ == "__main__":
    unittest.main()
