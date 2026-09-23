"""Independent structural checks for B2 synthetic tuning and holdout suites."""

from __future__ import annotations

from datetime import timedelta
import hashlib
import json
from typing import Any

from stage1.simulation import B2_SCENARIO_NAMES, SyntheticSuite
from stage1.observability import ObservabilityPolicy, audit_channel
from stage1.profiles import fit_behavior_profiles


def _scenario_events(suite: SyntheticSuite, scenario_id: str):
    return tuple(event for event in suite.events if event.source == f"synthetic:{scenario_id}")


def _prefix_digest(suite: SyntheticSuite, scenario_id: str, cutoff) -> str:
    records = [
        event.to_record()
        for event in _scenario_events(suite, scenario_id)
        if event.timestamp < cutoff
    ]
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _suite_checks(suite: SyntheticSuite) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    manifest = suite.manifest()
    truth_by_name = {item.scenario_name: item for item in suite.truth}
    checks = {
        "all_b2_scenarios_present": tuple(item.scenario_name for item in suite.truth)
        == B2_SCENARIO_NAMES,
        "unique_scenario_ids": len({item.scenario_id for item in suite.truth}) == len(suite.truth),
        "at_least_20_channels_selected": len(manifest["validation_channels"]) == 20,
        "year_2021_excluded": manifest["excluded_source_years"] == [2021]
        and all(event.timestamp.year != 2021 for event in suite.events),
        "manifest_event_hash_is_stable": manifest["events_sha256"]
        == suite.manifest()["events_sha256"],
        "both_numeric_and_state_data_exist": any(
            event.numeric_value is not None for event in suite.events
        )
        and any(event.numeric_value is None for event in suite.events),
    }
    details = []
    for truth in suite.truth:
        events = _scenario_events(suite, truth.scenario_id)
        prefix = tuple(event for event in events if event.timestamp < truth.intervention_start)
        causal_prefix_digest = _prefix_digest(suite, truth.scenario_id, truth.intervention_start)
        details.append(
            {
                "scenario_id": truth.scenario_id,
                "scenario_name": truth.scenario_name,
                "events": len(events),
                "channels": len(truth.channel_ids),
                "clean_prefix_seconds": int(
                    (
                        truth.intervention_start - min(event.timestamp for event in prefix)
                    ).total_seconds()
                )
                if prefix
                else 0,
                "causal_prefix_sha256": causal_prefix_digest,
                "passed": bool(events)
                and bool(prefix)
                and truth.intervention_start - min(event.timestamp for event in prefix)
                >= timedelta(days=7)
                and all(event.channel_id in truth.channel_ids for event in events),
            }
        )
    checks["every_scenario_has_clean_causal_prefix"] = all(item["passed"] for item in details)

    stuck = truth_by_name["numeric_stuck"]
    stuck_values = {
        event.numeric_value
        for event in _scenario_events(suite, stuck.scenario_id)
        if stuck.intervention_start <= event.timestamp < stuck.end
    }
    checks["stuck_keeps_messages_at_one_value"] = len(stuck_values) == 1

    dropout = truth_by_name["known_cadence_dropout"]
    dropout_events = _scenario_events(suite, dropout.scenario_id)
    checks["known_dropout_contains_no_fabricated_zero"] = not any(
        dropout.intervention_start <= event.timestamp < dropout.end for event in dropout_events
    ) and all(event.raw_value != "0" for event in dropout_events)

    degraded = truth_by_name["degraded_communication"]
    before_count = sum(
        degraded.intervention_start - timedelta(hours=24)
        <= event.timestamp
        < degraded.intervention_start
        for event in _scenario_events(suite, degraded.scenario_id)
    )
    during_count = sum(
        degraded.intervention_start <= event.timestamp < degraded.end
        for event in _scenario_events(suite, degraded.scenario_id)
    )
    checks["communication_degradation_reduces_observations"] = (
        before_count > 0 and 0 < during_count < before_count * 3
    )

    mixed = truth_by_name["mixed_numeric_state"]
    mixed_events = _scenario_events(suite, mixed.scenario_id)
    checks["mixed_channel_preserves_both_branches"] = any(
        event.numeric_value is not None for event in mixed_events
    ) and any(event.numeric_value is None for event in mixed_events)

    export_gap = truth_by_name["export_gap_control"]
    export_events = _scenario_events(suite, export_gap.scenario_id)
    checks["export_gap_is_shared_missingness"] = all(
        not any(
            export_gap.intervention_start <= event.timestamp < export_gap.end
            for event in export_events
            if event.channel_id == channel_id
        )
        for channel_id in export_gap.channel_ids
    )

    environment = truth_by_name["common_environment_control"]
    checks["common_environment_is_control"] = (
        environment.label == "control"
        and environment.failure_point is None
        and environment.cause_hypothesis == "common_environment"
    )
    feature_checks = []
    scenario_by_channel = {
        channel_id: truth for truth in suite.truth for channel_id in truth.channel_ids
    }
    for channel_id in manifest["validation_channels"]:
        truth = scenario_by_channel[channel_id]
        channel_events = tuple(event for event in suite.events if event.channel_id == channel_id)
        past = tuple(
            event for event in channel_events if event.timestamp <= truth.intervention_start
        )
        cadence = (
            timedelta(seconds=truth.expected_cadence_seconds)
            if truth.expected_cadence_seconds is not None
            else None
        )
        policy = ObservabilityPolicy(expected_cadence=cadence, minimum_window_events=1)
        audit_from_prefix = audit_channel(channel_id, past, truth.intervention_start, policy)
        audit_with_future = audit_channel(
            channel_id, channel_events, truth.intervention_start, policy
        )
        profile_from_prefix = fit_behavior_profiles(
            past, truth.intervention_start, min_events=1, min_numeric=1
        )[channel_id]
        profile_with_future = fit_behavior_profiles(
            channel_events, truth.intervention_start, min_events=1, min_numeric=1
        )[channel_id]
        feature_checks.append(
            {
                "channel_id": channel_id,
                "scenario_id": truth.scenario_id,
                "sensor_type": truth.sensor_type,
                "observability_unchanged_by_future": audit_from_prefix == audit_with_future,
                "profile_unchanged_by_future": profile_from_prefix == profile_with_future,
                "profile_has_only_past": profile_from_prefix.last_observation
                < truth.intervention_start,
            }
        )
    checks["20_channel_causal_feature_contract"] = len(feature_checks) == 20 and all(
        all(value for key, value in item.items() if key.endswith("future") or key.endswith("past"))
        for item in feature_checks
    )
    for item in details:
        item["feature_contract_channels"] = [
            check for check in feature_checks if check["scenario_id"] == item["scenario_id"]
        ]
    return checks, details


def validate_b2_suites(tuning: SyntheticSuite, holdout: SyntheticSuite) -> dict[str, Any]:
    """Validate suite content and independence without running any detector."""

    if tuning.name != "tuning" or holdout.name != "synthetic_holdout":
        raise ValueError("expected tuning and synthetic_holdout suites")
    tuning_checks, tuning_details = _suite_checks(tuning)
    holdout_checks, holdout_details = _suite_checks(holdout)
    tuning_manifest, holdout_manifest = tuning.manifest(), holdout.manifest()
    cross_checks = {
        "scenario_ids_disjoint": {item.scenario_id for item in tuning.truth}.isdisjoint(
            item.scenario_id for item in holdout.truth
        ),
        "channel_ids_disjoint": {event.channel_id for event in tuning.events}.isdisjoint(
            event.channel_id for event in holdout.events
        ),
        "background_ids_disjoint": {
            item["background_id"] for item in tuning_manifest["scenarios"]
        }.isdisjoint(item["background_id"] for item in holdout_manifest["scenarios"]),
        "parent_background_ids_disjoint": {
            item["parent_background_id"] for item in tuning_manifest["scenarios"]
        }.isdisjoint(item["parent_background_id"] for item in holdout_manifest["scenarios"]),
        "event_content_differs": tuning_manifest["events_sha256"]
        != holdout_manifest["events_sha256"],
        "segments_do_not_overlap": max(event.timestamp for event in tuning.events)
        < min(event.timestamp for event in holdout.events),
    }
    passed = all((*tuning_checks.values(), *holdout_checks.values(), *cross_checks.values()))
    return {
        "validation_version": "b2-validation-v1",
        "passed": passed,
        "excluded_source_years": [2021],
        "tuning": {"checks": tuning_checks, "scenarios": tuning_details},
        "synthetic_holdout": {"checks": holdout_checks, "scenarios": holdout_details},
        "cross_suite_checks": cross_checks,
    }
