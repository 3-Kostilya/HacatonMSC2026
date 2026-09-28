"""Frozen, deterministic synthetic scenarios for stage-one evaluation.

The generator deliberately creates observations, not physical diagnoses.  Every
scenario has a clean causal prefix and a separate truth record.  The missingness
case removes observations but never inserts a synthetic "failure" value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Iterator

from stage1.contracts import NormalizedEvent


CONFIG_PATH = Path(__file__).with_name("config") / "simulation_parameters.json"
SCENARIO_NAMES = (
    "numeric_gradual_drift",
    "numeric_variance_increase",
    "numeric_level_shift",
    "numeric_single_spike_control",
    "discrete_rapid_switching",
    "discrete_ordinary_cycle_control",
    "single_alarm_control",
    "missingness_unknown_control",
    "common_outage_context",
)

B2_ADDITIONAL_SCENARIO_NAMES = (
    "numeric_stuck",
    "numeric_seasonality_control",
    "numeric_heavy_tail_control",
    "known_cadence_dropout",
    "degraded_communication",
    "export_gap_control",
    "common_environment_control",
    "mixed_numeric_state",
)
B2_SCENARIO_NAMES = (*SCENARIO_NAMES, *B2_ADDITIONAL_SCENARIO_NAMES)
EXCLUDED_SOURCE_YEARS = (2021,)


@dataclass(frozen=True, slots=True)
class ScenarioTruth:
    """Machine-readable expected behavior for one frozen scenario."""

    scenario_id: str
    scenario_name: str
    suite: str
    channel_ids: tuple[str, ...]
    sensor_type: str
    intervention_start: datetime
    failure_point: datetime | None
    end: datetime
    label: str
    expected_detector_behavior: str
    detector_applicability: str
    expected_cadence_seconds: int | None
    cause_hypothesis: str
    notes: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        for field in ("intervention_start", "failure_point", "end"):
            value = getattr(self, field)
            record[field] = value.isoformat(sep=" ") if value is not None else None
        record["channel_ids"] = list(self.channel_ids)
        record["notes"] = list(self.notes)
        return record


@dataclass(frozen=True, slots=True)
class SyntheticSuite:
    """Events and truth created from one immutable suite configuration."""

    name: str
    simulation_version: str
    seed: int
    parameters: dict[str, Any]
    events: tuple[NormalizedEvent, ...]
    truth: tuple[ScenarioTruth, ...]

    def manifest(self) -> dict[str, Any]:
        event_text = _events_text(self.events)
        scenarios = []
        for item in self.truth:
            record = item.to_record()
            scenario_events = tuple(
                event for event in self.events if event.source == f"synthetic:{item.scenario_id}"
            )
            record.update(
                {
                    "background_id": f"{self.name}:{item.scenario_name}:background",
                    "parent_background_id": _parent_background_id(self.name, item.scenario_name),
                    "source_channel_ids": list(item.channel_ids),
                    "source_interval": {
                        "start": min(event.timestamp for event in scenario_events).isoformat(
                            sep=" "
                        ),
                        "end": max(event.timestamp for event in scenario_events).isoformat(sep=" "),
                    },
                    "scenario_seed": _scenario_seed(self.seed, self.name, item.scenario_name),
                    "event_sha256": _event_digest(scenario_events),
                }
            )
            scenarios.append(record)
        return {
            "simulation_version": self.simulation_version,
            "suite": self.name,
            "seed": self.seed,
            "excluded_source_years": list(EXCLUDED_SOURCE_YEARS),
            "parameters": self.parameters,
            "event_count": len(self.events),
            "events_sha256": hashlib.sha256(event_text.encode("utf-8")).hexdigest(),
            "validation_channels": _validation_channels(self.events),
            "scenarios": scenarios,
        }


def _event_json(event: NormalizedEvent) -> str:
    return json.dumps(event.to_record(), ensure_ascii=False, sort_keys=True)


def _events_text(events: Iterable[NormalizedEvent]) -> str:
    return "".join(_event_json(event) + "\n" for event in events)


def _event_digest(events: Iterable[NormalizedEvent]) -> str:
    return hashlib.sha256(_events_text(events).encode("utf-8")).hexdigest()


def _parent_background_id(suite: str, scenario_name: str) -> str:
    if scenario_name.startswith("numeric_"):
        family = "numeric"
    elif scenario_name in {
        "discrete_rapid_switching",
        "discrete_ordinary_cycle_control",
        "single_alarm_control",
    }:
        family = "discrete"
    elif scenario_name in {"known_cadence_dropout", "degraded_communication"}:
        family = "known-cadence"
    elif scenario_name in {"export_gap_control", "common_outage_context"}:
        family = "multi-channel-observability"
    else:
        family = scenario_name
    return f"{suite}:{family}:parent"


def _validation_channels(events: Iterable[NormalizedEvent], count: int = 20) -> list[str]:
    """Choose a stable 20-channel contract sample without looking at model output."""

    by_type: dict[str, list[str]] = {}
    for event in events:
        by_type.setdefault(event.sensor_type, []).append(event.channel_id)
    selected = []
    for sensor_type in sorted(by_type):
        selected.append(sorted(set(by_type[sensor_type]))[0])
    remaining = sorted({event.channel_id for event in events} - set(selected))
    return (selected + remaining)[:count]


def load_simulation_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and minimally validate the parameters frozen before evaluation."""

    config = json.loads(path.read_text(encoding="utf-8"))
    if config["seed"] != 260919:
        raise ValueError("simulation seed must match evaluation_protocol.json")
    if config["minimum_clean_prefix_seconds"] < 604800:
        raise ValueError("clean prefix must be at least seven days")
    for suite in ("tuning", "synthetic_holdout"):
        if suite not in config["suites"]:
            raise ValueError(f"missing suite: {suite}")
        settings = config["suites"][suite]
        if datetime.fromisoformat(settings["segment_start"]).year in EXCLUDED_SOURCE_YEARS:
            raise ValueError(f"{suite}: excluded source year used by synthetic segment")
        for parameter, bounds in config["declared_ranges"].items():
            value = settings["parameters"].get(parameter)
            if value is None or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"{suite}: {parameter} is outside its declared range")
    return config


def _scenario_seed(seed: int, suite: str, scenario_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{suite}:{scenario_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _event(
    *,
    channel_id: str,
    timestamp: datetime,
    raw_value: str,
    sensor_type: str,
    scenario_id: str,
    index: int,
    numeric_value: float | None = None,
    alarm: bool = False,
    object_id: str | None = None,
) -> NormalizedEvent:
    return NormalizedEvent(
        channel_id=channel_id,
        timestamp=timestamp,
        raw_value=raw_value,
        alarm=alarm,
        sensor_type=sensor_type,
        source=f"synthetic:{scenario_id}",
        event_id=f"{scenario_id}:{index:05d}",
        object_id=object_id,
        numeric_value=numeric_value,
        quality_flags=("synthetic_observation",),
    )


def _numeric_baseline(
    channel_id: str,
    start: datetime,
    end: datetime,
    cadence: timedelta,
    rng: random.Random,
    scenario_id: str,
) -> list[NormalizedEvent]:
    events: list[NormalizedEvent] = []
    timestamp = start
    value = 20.0
    index = 0
    while timestamp < end:
        value = 20.0 + 0.75 * (value - 20.0) + rng.gauss(0.0, 0.35)
        events.append(
            _event(
                channel_id=channel_id,
                timestamp=timestamp,
                raw_value=f"{value:.4f}",
                numeric_value=value,
                sensor_type="Датчик температуры",
                scenario_id=scenario_id,
                index=index,
            )
        )
        timestamp += cadence
        index += 1
    return events


def _numeric_scenario(
    name: str,
    suite: str,
    start: datetime,
    prefix: timedelta,
    cadence: timedelta,
    parameters: dict[str, Any],
    rng: random.Random,
) -> tuple[list[NormalizedEvent], ScenarioTruth]:
    scenario_id = f"{suite}:{name}:001"
    channel_id = f"sim-{suite}-{name}"
    intervention = start + prefix
    end = intervention + timedelta(days=3)
    baseline = _numeric_baseline(channel_id, start, end, cadence, rng, scenario_id)
    prefix_values = [event.numeric_value for event in baseline if event.timestamp < intervention]
    assert prefix_values and all(value is not None for value in prefix_values)
    mean = sum(value for value in prefix_values if value is not None) / len(prefix_values)
    variance = sum((value - mean) ** 2 for value in prefix_values if value is not None)
    sigma = max((variance / len(prefix_values)) ** 0.5, 0.1)

    changed: list[NormalizedEvent] = []
    post_index = 0
    for index, event in enumerate(baseline):
        value = event.numeric_value
        assert value is not None
        if event.timestamp >= intervention:
            if name == "numeric_gradual_drift":
                value += post_index * parameters["drift_per_step_sigma"] * sigma
            elif name == "numeric_variance_increase":
                value = mean + (value - mean) * parameters["variance_multiplier"]
            elif name == "numeric_level_shift":
                value += parameters["level_shift_sigma"] * sigma
            elif name == "numeric_single_spike_control" and post_index == 4:
                value += parameters["spike_sigma"] * sigma
            post_index += 1
        changed.append(
            _event(
                channel_id=channel_id,
                timestamp=event.timestamp,
                raw_value=f"{value:.4f}",
                numeric_value=value,
                sensor_type=event.sensor_type,
                scenario_id=scenario_id,
                index=index,
            )
        )
    control = name == "numeric_single_spike_control"
    truth = ScenarioTruth(
        scenario_id=scenario_id,
        scenario_name=name,
        suite=suite,
        channel_ids=(channel_id,),
        sensor_type="Датчик температуры",
        intervention_start=intervention,
        failure_point=None if control else intervention,
        end=end,
        label="control" if control else "positive",
        expected_detector_behavior="no_candidate" if control else "candidate",
        detector_applicability="numeric",
        expected_cadence_seconds=int(cadence.total_seconds()),
        cause_hypothesis="unknown",
        notes=("effect is expressed in baseline sigma, not physical units",),
    )
    return changed, truth


def _discrete_scenario(
    name: str,
    suite: str,
    start: datetime,
    prefix: timedelta,
    cadence: timedelta,
    parameters: dict[str, Any],
) -> tuple[list[NormalizedEvent], ScenarioTruth]:
    scenario_id = f"{suite}:{name}:001"
    channel_id = f"sim-{suite}-{name}"
    intervention = start + prefix
    end = intervention + timedelta(days=2)
    events: list[NormalizedEvent] = []
    timestamp = start
    index = 0
    sensor_type = "Датчик дыма" if name == "single_alarm_control" else "Состояние насоса"
    while timestamp < intervention:
        state = (
            "Норма"
            if name == "single_alarm_control"
            else "Включен"
            if index % 8 in (2, 3)
            else "Выключен"
        )
        events.append(
            _event(
                channel_id=channel_id,
                timestamp=timestamp,
                raw_value=state,
                sensor_type=sensor_type,
                scenario_id=scenario_id,
                index=index,
            )
        )
        index += 1
        timestamp += cadence

    control = name != "discrete_rapid_switching"
    if name == "discrete_rapid_switching":
        step = timedelta(seconds=parameters["rapid_switch_seconds"])
        state = "Выключен"
        while timestamp < end:
            state = "Включен" if state == "Выключен" else "Выключен"
            events.append(
                _event(
                    channel_id=channel_id,
                    timestamp=timestamp,
                    raw_value=state,
                    sensor_type="Состояние насоса",
                    scenario_id=scenario_id,
                    index=index,
                )
            )
            index += 1
            timestamp += step
    elif name == "discrete_ordinary_cycle_control":
        while timestamp < end:
            state = "Включен" if index % 8 in (2, 3) else "Выключен"
            events.append(
                _event(
                    channel_id=channel_id,
                    timestamp=timestamp,
                    raw_value=state,
                    sensor_type="Состояние насоса",
                    scenario_id=scenario_id,
                    index=index,
                )
            )
            index += 1
            timestamp += cadence
    else:
        while timestamp < end:
            alarm = timestamp == intervention
            events.append(
                _event(
                    channel_id=channel_id,
                    timestamp=timestamp,
                    raw_value="Тревога" if alarm else "Норма",
                    alarm=alarm,
                    sensor_type="Датчик дыма",
                    scenario_id=scenario_id,
                    index=index,
                )
            )
            index += 1
            timestamp += cadence

    truth = ScenarioTruth(
        scenario_id=scenario_id,
        scenario_name=name,
        suite=suite,
        channel_ids=(channel_id,),
        sensor_type="Состояние насоса" if name != "single_alarm_control" else "Датчик дыма",
        intervention_start=intervention,
        failure_point=intervention if not control else None,
        end=end,
        label="control" if control else "positive",
        expected_detector_behavior="no_candidate" if control else "candidate",
        detector_applicability="discrete",
        expected_cadence_seconds=int(cadence.total_seconds()),
        cause_hypothesis="unknown",
        notes=("alarm flag is an observation, never a ready-made failure label",),
    )
    return events, truth


def _missingness_scenario(
    suite: str, start: datetime, prefix: timedelta, parameters: dict[str, Any]
) -> tuple[list[NormalizedEvent], ScenarioTruth]:
    name = "missingness_unknown_control"
    scenario_id = f"{suite}:{name}:001"
    channel_id = f"sim-{suite}-{name}"
    intervention = start + prefix
    end = intervention + timedelta(seconds=parameters["gap_seconds"])
    # Irregular observations make cadence explicitly unknown.  The gap contains
    # no generated rows and therefore cannot masquerade as a failure message.
    offsets = (0, 5, 13, 22, 37, 55, 81, 116, 160, 213)
    events = [
        _event(
            channel_id=channel_id,
            timestamp=start + timedelta(hours=hours),
            raw_value="Норма",
            sensor_type="КД Дверь",
            scenario_id=scenario_id,
            index=index,
        )
        for index, hours in enumerate(offsets)
        if start + timedelta(hours=hours) < intervention
    ]
    events.append(
        _event(
            channel_id=channel_id,
            timestamp=end + timedelta(hours=11),
            raw_value="Норма",
            sensor_type="КД Дверь",
            scenario_id=scenario_id,
            index=len(events),
        )
    )
    truth = ScenarioTruth(
        scenario_id=scenario_id,
        scenario_name=name,
        suite=suite,
        channel_ids=(channel_id,),
        sensor_type="КД Дверь",
        intervention_start=intervention,
        failure_point=None,
        end=end,
        label="control",
        expected_detector_behavior="unknown",
        detector_applicability="observability_only_cadence_unknown",
        expected_cadence_seconds=None,
        cause_hypothesis="unknown",
        notes=(
            "silence is not modeled as failure",
            "candidate detection is not applicable because cadence is unknown",
        ),
    )
    return events, truth


def _outage_scenario(
    suite: str,
    start: datetime,
    prefix: timedelta,
    cadence: timedelta,
    parameters: dict[str, Any],
) -> tuple[list[NormalizedEvent], ScenarioTruth]:
    name = "common_outage_context"
    scenario_id = f"{suite}:{name}:001"
    intervention = start + prefix
    end = intervention + timedelta(hours=8)
    count = parameters["outage_channel_count"]
    channel_ids = tuple(f"sim-{suite}-outage-{index + 1}" for index in range(count))
    object_id = f"sim-{suite}-explicit-outage-object"
    events: list[NormalizedEvent] = []
    index = 0
    for channel_number, channel_id in enumerate(channel_ids):
        # Normal observations are deliberately staggered outside the context
        # detector's two-minute window.  Only the outage is synchronous.
        timestamp = start + timedelta(minutes=5 * channel_number)
        while timestamp < intervention:
            events.append(
                _event(
                    channel_id=channel_id,
                    timestamp=timestamp,
                    raw_value="Связь есть",
                    sensor_type="Состояние фазы",
                    scenario_id=scenario_id,
                    index=index,
                    object_id=object_id,
                )
            )
            index += 1
            timestamp += cadence
        outage_observations = (
            (intervention, "Нет связи"),
            (intervention + timedelta(minutes=1), "Нет связи"),
            (end, "Связь есть"),
        )
        for timestamp, value in outage_observations:
            events.append(
                _event(
                    channel_id=channel_id,
                    timestamp=timestamp,
                    raw_value=value,
                    sensor_type="Состояние фазы",
                    scenario_id=scenario_id,
                    index=index,
                    object_id=object_id,
                )
            )
            index += 1
    truth = ScenarioTruth(
        scenario_id=scenario_id,
        scenario_name=name,
        suite=suite,
        channel_ids=channel_ids,
        sensor_type="Состояние фазы",
        intervention_start=intervention,
        failure_point=intervention,
        end=end,
        label="positive",
        expected_detector_behavior="single_common_context_candidate",
        detector_applicability="context",
        expected_cadence_seconds=int(cadence.total_seconds()),
        cause_hypothesis="common_outage",
        notes=(
            f"explicit synthetic relation: shared object_id={object_id}",
            "do not duplicate this into independent local failure diagnoses",
        ),
    )
    return events, truth


def _b2_numeric_scenario(
    name: str,
    suite: str,
    start: datetime,
    prefix: timedelta,
    cadence: timedelta,
    parameters: dict[str, Any],
    rng: random.Random,
) -> tuple[list[NormalizedEvent], ScenarioTruth]:
    scenario_id = f"{suite}:{name}:001"
    channel_id = f"sim-{suite}-{name}"
    intervention = start + prefix
    duration = timedelta(hours=parameters["b2_numeric_duration_hours"])
    end = intervention + duration
    baseline = _numeric_baseline(channel_id, start, end + cadence, cadence, rng, scenario_id)
    clean = [event.numeric_value for event in baseline if event.timestamp < intervention]
    assert clean and all(value is not None for value in clean)
    mean = sum(value for value in clean if value is not None) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean if value is not None) / len(clean)
    sigma = max(variance**0.5, 0.1)
    stuck_value = next(
        event.numeric_value for event in reversed(baseline) if event.timestamp < intervention
    )
    changed = []
    for index, event in enumerate(baseline):
        value = event.numeric_value
        assert value is not None and stuck_value is not None
        if name == "numeric_stuck" and intervention <= event.timestamp < end:
            value = stuck_value
        elif name == "numeric_seasonality_control":
            elapsed_hours = (event.timestamp - start).total_seconds() / 3600
            value = mean + parameters["seasonal_amplitude_sigma"] * sigma * math.sin(
                2 * math.pi * elapsed_hours / 24
            )
        elif name == "numeric_heavy_tail_control" and index % 37 == 0:
            # Symmetric, deterministic rare tails exist before and after the evaluation boundary.
            value = mean + (parameters["heavy_tail_sigma"] * sigma * (-1 if index % 74 else 1))
        changed.append(
            _event(
                channel_id=channel_id,
                timestamp=event.timestamp,
                raw_value=f"{value:.4f}",
                numeric_value=value,
                sensor_type="Датчик температуры",
                scenario_id=scenario_id,
                index=index,
            )
        )
    control = name.endswith("_control")
    notes = (
        (
            "constant value while messages continue; expected variability exists in clean prefix",
            "stuck is distinct from communication dropout",
        )
        if name == "numeric_stuck"
        else (
            "background pattern exists on both sides of the evaluation boundary",
            "control must remain in the false-warning denominator",
        )
    )
    return changed, ScenarioTruth(
        scenario_id=scenario_id,
        scenario_name=name,
        suite=suite,
        channel_ids=(channel_id,),
        sensor_type="Датчик температуры",
        intervention_start=intervention,
        failure_point=None if control else intervention,
        end=end,
        label="control" if control else "positive",
        expected_detector_behavior="no_candidate" if control else "research_candidate",
        detector_applicability="b2_numeric_control" if control else "b2_numeric_stuck",
        expected_cadence_seconds=int(cadence.total_seconds()),
        cause_hypothesis="unknown",
        notes=notes,
    )


def _b2_cadence_scenario(
    name: str,
    suite: str,
    start: datetime,
    prefix: timedelta,
    cadence: timedelta,
    parameters: dict[str, Any],
) -> tuple[list[NormalizedEvent], ScenarioTruth]:
    scenario_id = f"{suite}:{name}:001"
    channel_id = f"sim-{suite}-{name}"
    intervention = start + prefix
    duration = timedelta(hours=parameters[f"{name}_hours"])
    end = intervention + duration
    events = []
    timestamp = start
    index = 0
    while timestamp <= end + cadence:
        in_effect = intervention <= timestamp < end
        keep = not in_effect
        if name == "degraded_communication" and in_effect:
            elapsed_steps = int((timestamp - intervention) / cadence)
            keep = elapsed_steps % parameters["degradation_keep_every"] == 0
        if keep:
            events.append(
                _event(
                    channel_id=channel_id,
                    timestamp=timestamp,
                    raw_value="Связь есть",
                    sensor_type="Состояние фазы",
                    scenario_id=scenario_id,
                    index=index,
                )
            )
            index += 1
        timestamp += cadence
    return events, ScenarioTruth(
        scenario_id=scenario_id,
        scenario_name=name,
        suite=suite,
        channel_ids=(channel_id,),
        sensor_type="Состояние фазы",
        intervention_start=intervention,
        failure_point=intervention,
        end=end,
        label="positive",
        expected_detector_behavior="observability_unknown",
        detector_applicability="b2_observability_known_cadence",
        expected_cadence_seconds=int(cadence.total_seconds()),
        cause_hypothesis="communication_quality",
        notes=(
            "absence is represented by missing observations, never by a zero value",
            "expected cadence is explicit and fixed before evaluation",
        ),
    )


def _b2_multi_channel_scenario(
    name: str,
    suite: str,
    start: datetime,
    prefix: timedelta,
    cadence: timedelta,
    parameters: dict[str, Any],
) -> tuple[list[NormalizedEvent], ScenarioTruth]:
    scenario_id = f"{suite}:{name}:001"
    intervention = start + prefix
    duration = timedelta(hours=parameters["multi_channel_effect_hours"])
    end = intervention + duration
    count = parameters[
        "export_gap_channel_count" if name == "export_gap_control" else "environment_channel_count"
    ]
    channel_ids = tuple(f"sim-{suite}-{name}-{index + 1}" for index in range(count))
    events = []
    index = 0
    object_id = f"sim-{suite}-environment-object" if name == "common_environment_control" else None
    for channel_number, channel_id in enumerate(channel_ids):
        timestamp = start
        while timestamp <= end + cadence:
            in_effect = intervention <= timestamp < end
            if name == "export_gap_control" and in_effect:
                timestamp += cadence
                continue
            if name == "common_environment_control":
                value = 20.0 + channel_number * 0.1 + (4.0 if in_effect else 0.0)
                raw_value, numeric_value, sensor_type = f"{value:.2f}", value, "Датчик температуры"
            else:
                raw_value, numeric_value, sensor_type = "Норма", None, "КД Дверь"
            events.append(
                _event(
                    channel_id=channel_id,
                    timestamp=timestamp,
                    raw_value=raw_value,
                    numeric_value=numeric_value,
                    sensor_type=sensor_type,
                    scenario_id=scenario_id,
                    index=index,
                    object_id=object_id,
                )
            )
            index += 1
            timestamp += cadence
    expected = "unknown" if name == "export_gap_control" else "no_local_failure_candidate"
    return events, ScenarioTruth(
        scenario_id=scenario_id,
        scenario_name=name,
        suite=suite,
        channel_ids=channel_ids,
        sensor_type="КД Дверь" if name == "export_gap_control" else "Датчик температуры",
        intervention_start=intervention,
        failure_point=None,
        end=end,
        label="control",
        expected_detector_behavior=expected,
        detector_applicability=(
            "b2_export_gap_control" if name == "export_gap_control" else "b2_context_control"
        ),
        expected_cadence_seconds=int(cadence.total_seconds()),
        cause_hypothesis="export_gap" if name == "export_gap_control" else "common_environment",
        notes=(
            "multi-channel effect must not be multiplied into independent physical failures",
            "object relation is explicit only for the common-environment control",
        ),
    )


def _b2_mixed_scenario(
    suite: str,
    start: datetime,
    prefix: timedelta,
    cadence: timedelta,
    parameters: dict[str, Any],
) -> tuple[list[NormalizedEvent], ScenarioTruth]:
    name = "mixed_numeric_state"
    scenario_id = f"{suite}:{name}:001"
    channel_id = f"sim-{suite}-{name}"
    intervention = start + prefix
    end = intervention + timedelta(hours=parameters["mixed_duration_hours"])
    events = []
    timestamp = start
    index = 0
    while timestamp <= end:
        numeric = index % 2 == 0
        if numeric:
            value = 220.0 + (parameters["mixed_shift"] if timestamp >= intervention else 0.0)
            raw_value, numeric_value = f"{value:.1f}", value
        else:
            raw_value, numeric_value = ("Батарея" if timestamp >= intervention else "Сеть"), None
        events.append(
            _event(
                channel_id=channel_id,
                timestamp=timestamp,
                raw_value=raw_value,
                numeric_value=numeric_value,
                sensor_type="ИБП",
                scenario_id=scenario_id,
                index=index,
            )
        )
        index += 1
        timestamp += cadence
    return events, ScenarioTruth(
        scenario_id=scenario_id,
        scenario_name=name,
        suite=suite,
        channel_ids=(channel_id,),
        sensor_type="ИБП",
        intervention_start=intervention,
        failure_point=intervention,
        end=end,
        label="positive",
        expected_detector_behavior="preserve_both_branches",
        detector_applicability="b2_mixed",
        expected_cadence_seconds=int(cadence.total_seconds()),
        cause_hypothesis="unknown",
        notes=("numeric and state observations share one channel and must remain distinct",),
    )


def build_suite(name: str, config_path: Path = CONFIG_PATH) -> SyntheticSuite:
    """Build the frozen tuning or holdout suite without reading detector output."""

    config = load_simulation_config(config_path)
    if name not in config["suites"]:
        raise ValueError(f"unknown suite: {name}")
    settings = config["suites"][name]
    parameters = dict(settings["parameters"])
    start = datetime.fromisoformat(settings["segment_start"])
    prefix = timedelta(seconds=config["minimum_clean_prefix_seconds"])
    numeric_cadence = timedelta(seconds=settings["numeric_cadence_seconds"])
    discrete_cadence = timedelta(seconds=settings["discrete_cadence_seconds"])
    all_events: list[NormalizedEvent] = []
    truth: list[ScenarioTruth] = []

    for scenario_name in SCENARIO_NAMES[:4]:
        # All numeric interventions start from the same frozen clean trajectory.
        # Reinitialising this RNG for each case makes prefix equality directly testable.
        rng = random.Random(_scenario_seed(config["seed"], name, "numeric_baseline"))
        events, item = _numeric_scenario(
            scenario_name, name, start, prefix, numeric_cadence, parameters, rng
        )
        all_events.extend(events)
        truth.append(item)
    for scenario_name in SCENARIO_NAMES[4:7]:
        events, item = _discrete_scenario(
            scenario_name, name, start, prefix, discrete_cadence, parameters
        )
        all_events.extend(events)
        truth.append(item)
    events, item = _missingness_scenario(name, start, prefix, parameters)
    all_events.extend(events)
    truth.append(item)
    events, item = _outage_scenario(name, start, prefix, discrete_cadence, parameters)
    all_events.extend(events)
    truth.append(item)
    all_events.sort(key=lambda event: (event.timestamp, event.channel_id, event.event_id or ""))
    return SyntheticSuite(
        name=name,
        simulation_version=config["simulation_version"],
        seed=config["seed"],
        parameters={
            "segment_start": settings["segment_start"],
            "minimum_clean_prefix_seconds": config["minimum_clean_prefix_seconds"],
            "numeric_cadence_seconds": settings["numeric_cadence_seconds"],
            "discrete_cadence_seconds": settings["discrete_cadence_seconds"],
            **parameters,
        },
        events=tuple(all_events),
        truth=tuple(truth),
    )


def build_b2_suite(name: str, config_path: Path = CONFIG_PATH) -> SyntheticSuite:
    """Build B2's expanded tuning or holdout suite without detector feedback."""

    base = build_suite(name, config_path)
    config = load_simulation_config(config_path)
    settings = config["suites"][name]
    parameters = dict(settings["parameters"])
    start = datetime.fromisoformat(settings["segment_start"])
    prefix = timedelta(seconds=config["minimum_clean_prefix_seconds"])
    numeric_cadence = timedelta(seconds=settings["numeric_cadence_seconds"])
    discrete_cadence = timedelta(seconds=settings["discrete_cadence_seconds"])
    events = list(base.events)
    truth = list(base.truth)

    for scenario_name in (
        "numeric_stuck",
        "numeric_seasonality_control",
        "numeric_heavy_tail_control",
    ):
        rng = random.Random(_scenario_seed(config["seed"], name, scenario_name))
        scenario_events, item = _b2_numeric_scenario(
            scenario_name, name, start, prefix, numeric_cadence, parameters, rng
        )
        events.extend(scenario_events)
        truth.append(item)
    for scenario_name in ("known_cadence_dropout", "degraded_communication"):
        scenario_events, item = _b2_cadence_scenario(
            scenario_name, name, start, prefix, discrete_cadence, parameters
        )
        events.extend(scenario_events)
        truth.append(item)
    for scenario_name in ("export_gap_control", "common_environment_control"):
        scenario_events, item = _b2_multi_channel_scenario(
            scenario_name, name, start, prefix, discrete_cadence, parameters
        )
        events.extend(scenario_events)
        truth.append(item)
    scenario_events, item = _b2_mixed_scenario(name, start, prefix, discrete_cadence, parameters)
    events.extend(scenario_events)
    truth.append(item)
    events.sort(key=lambda event: (event.timestamp, event.channel_id, event.event_id or ""))
    return SyntheticSuite(
        name=name,
        simulation_version=config["simulation_version"],
        seed=config["seed"],
        parameters={**base.parameters, "b2_expanded": True, "excluded_source_years": [2021]},
        events=tuple(events),
        truth=tuple(truth),
    )


def iter_event_records(events: Iterable[NormalizedEvent]) -> Iterator[dict[str, Any]]:
    """Yield JSON-ready normalized event records."""

    for event in events:
        yield event.to_record()


def write_suite(suite: SyntheticSuite, output_directory: Path) -> tuple[Path, Path]:
    """Write normalized events as JSONL and truth/parameters as one manifest."""

    output_directory.mkdir(parents=True, exist_ok=True)
    events_path = output_directory / f"{suite.name}_events.jsonl"
    manifest_path = output_directory / f"{suite.name}_truth_manifest.json"
    events_text = _events_text(suite.events)
    events_path.write_text(events_text, encoding="utf-8", newline="\n")
    manifest_path.write_text(
        json.dumps(suite.manifest(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return events_path, manifest_path
