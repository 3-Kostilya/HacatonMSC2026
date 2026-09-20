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
        return {
            "simulation_version": self.simulation_version,
            "suite": self.name,
            "seed": self.seed,
            "parameters": self.parameters,
            "event_count": len(self.events),
            "scenarios": [item.to_record() for item in self.truth],
        }


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


def iter_event_records(events: Iterable[NormalizedEvent]) -> Iterator[dict[str, Any]]:
    """Yield JSON-ready normalized event records."""

    for event in events:
        yield event.to_record()


def write_suite(suite: SyntheticSuite, output_directory: Path) -> tuple[Path, Path]:
    """Write normalized events as JSONL and truth/parameters as one manifest."""

    output_directory.mkdir(parents=True, exist_ok=True)
    events_path = output_directory / f"{suite.name}_events.jsonl"
    manifest_path = output_directory / f"{suite.name}_truth_manifest.json"
    events_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in iter_event_records(suite.events)
    )
    events_path.write_text(events_text, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(suite.manifest(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return events_path, manifest_path
