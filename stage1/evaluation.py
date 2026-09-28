"""Frozen, detector-independent evaluation for stage-one warning episodes.

The evaluator deliberately knows nothing about how a warning was produced.  It
matches the public :class:`Episode` contract to truth records using the keys and
greedy rule frozen in ``stage1-eval-v1``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import math
from typing import Any, Iterable, Mapping, Sequence

from stage1.contracts import Decision, Episode


VALID_STATUSES = frozenset({"include", "unknown", "exclude"})
EXPECTED_SENSITIVITY_VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "numeric_mad_5_5": {"numeric.mad_multiplier": 5.5},
    "numeric_mad_6_5": {"numeric.mad_multiplier": 6.5},
    "numeric_confirmation_2": {"numeric.min_sustained": 2},
    "numeric_confirmation_4": {"numeric.min_sustained": 4},
    "discrete_transitions_3": {"discrete.transition_count": 3},
    "discrete_transitions_5": {"discrete.transition_count": 5},
    "discrete_window_8m": {"discrete.transition_window_seconds": 480},
    "discrete_window_12m": {"discrete.transition_window_seconds": 720},
    "without_numeric": {"ablation": "numeric"},
    "without_discrete": {"ablation": "discrete"},
    "without_context": {"ablation": "context"},
}


@dataclass(frozen=True, slots=True)
class TruthEpisode:
    episode_id: str
    channel_id: str
    scenario_id: str
    start_at: datetime
    end_at: datetime
    expected_cadence: timedelta | None
    sensor_type: str = "unknown"
    is_control: bool = False
    scope_channel_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.episode_id or not self.channel_id or not self.scenario_id:
            raise ValueError("truth identifiers must be non-empty")
        _validate_time(self.start_at)
        _validate_time(self.end_at)
        if self.end_at < self.start_at:
            raise ValueError("truth end_at cannot precede start_at")
        if self.expected_cadence is not None and self.expected_cadence <= timedelta(0):
            raise ValueError("expected_cadence must be positive")
        if any(not channel_id for channel_id in self.scope_channel_ids):
            raise ValueError("scope_channel_ids must be non-empty strings")

    @property
    def match_channel_ids(self) -> tuple[str, ...]:
        return self.scope_channel_ids or (self.channel_id,)


@dataclass(frozen=True, slots=True)
class EvaluationInterval:
    """One channel/scenario eligibility decision and its observed exposure."""

    channel_id: str
    scenario_id: str
    start_at: datetime
    end_at: datetime
    status: str
    sensor_type: str = "unknown"
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        _validate_time(self.start_at)
        _validate_time(self.end_at)
        if self.end_at <= self.start_at:
            raise ValueError("evaluation intervals must have positive duration")


@dataclass(frozen=True, slots=True)
class Match:
    truth_episode_id: str
    warning_episode_id: str
    channel_id: str
    scenario_id: str
    warning_at: datetime
    delay_seconds: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    protocol_version: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    delay_seconds_median: float | None
    delay_seconds_p90: float | None
    included_channel_days: float | None
    false_warnings_per_100_channel_days: float | None
    matches: tuple[Match, ...]
    unmatched_truth_ids: tuple[str, ...]
    unmatched_warning_ids: tuple[str, ...]
    ignored_warning_ids: tuple[str, ...]
    coverage: Mapping[str, Any]
    per_scenario: Mapping[str, Mapping[str, Any]]

    def to_record(self) -> dict[str, Any]:
        result = asdict(self)
        for match in result["matches"]:
            match["warning_at"] = match["warning_at"].isoformat(sep=" ")
        return result


def _validate_time(value: datetime) -> None:
    if not isinstance(value, datetime) or (
        value.tzinfo is not None and value.utcoffset() is not None
    ):
        raise ValueError("timestamps must use the journal's local naive time")


def _field(item: object, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _datetime(value: datetime | str, name: str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime or ISO timestamp")
    _validate_time(value)
    return value


def coerce_truth(item: TruthEpisode | Mapping[str, Any] | object) -> TruthEpisode:
    """Accept a dataclass, mapping, or object supplied by a scenario generator."""

    if isinstance(item, TruthEpisode):
        return item
    cadence = _field(item, "expected_cadence", _field(item, "expected_cadence_seconds"))
    if isinstance(cadence, (int, float)):
        cadence = timedelta(seconds=cadence)
    if cadence is not None and not isinstance(cadence, timedelta):
        raise ValueError("truth cadence must be timedelta or seconds")
    channel_ids = tuple(str(value) for value in _field(item, "channel_ids", ()))
    channel_id = _field(item, "channel_id", channel_ids[0] if channel_ids else None)
    scenario_id = str(_field(item, "scenario_id"))
    episode_id = _field(item, "episode_id", scenario_id)
    start = _field(item, "start_at", _field(item, "intervention_start"))
    end = _field(item, "end_at", _field(item, "end"))
    label = _field(item, "label")
    return TruthEpisode(
        episode_id=str(episode_id),
        channel_id=str(channel_id),
        scenario_id=scenario_id,
        start_at=_datetime(start, "start_at/intervention_start"),
        end_at=_datetime(end, "end_at/end"),
        expected_cadence=cadence,
        sensor_type=str(_field(item, "sensor_type", _field(item, "suite", "unknown"))),
        is_control=bool(_field(item, "is_control", _field(item, "control", label == "control"))),
        scope_channel_ids=channel_ids,
    )


def coerce_interval(item: EvaluationInterval | Mapping[str, Any] | object) -> EvaluationInterval:
    if isinstance(item, EvaluationInterval):
        return item
    return EvaluationInterval(
        channel_id=str(_field(item, "channel_id")),
        scenario_id=str(_field(item, "scenario_id")),
        start_at=_datetime(_field(item, "start_at"), "start_at"),
        end_at=_datetime(_field(item, "end_at"), "end_at"),
        status=str(_field(item, "status")),
        sensor_type=str(_field(item, "sensor_type", "unknown")),
        reasons=tuple(_field(item, "reasons", ())),
    )


def warning_scenario_id(episode: Episode) -> str:
    scenario_id = episode.metadata.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError(f"warning {episode.episode_id!r} has no metadata.scenario_id")
    return scenario_id


def _status_at(
    channel_id: str,
    scenario_id: str,
    moment: datetime,
    intervals: Sequence[EvaluationInterval],
) -> str | None:
    matching = [
        interval.status
        for interval in intervals
        if interval.channel_id == channel_id
        and interval.scenario_id == scenario_id
        and interval.start_at <= moment < interval.end_at
    ]
    if len(set(matching)) > 1:
        raise ValueError("overlapping eligibility intervals have conflicting statuses")
    return matching[0] if matching else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _scores(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _included_days(intervals: Sequence[EvaluationInterval]) -> float | None:
    included = [interval for interval in intervals if interval.status == "include"]
    if not intervals:
        return None
    # Union overlap within an experimental channel/scenario exposure.
    total = timedelta(0)
    by_scope: dict[tuple[str, str], list[tuple[datetime, datetime]]] = {}
    for interval in included:
        by_scope.setdefault((interval.channel_id, interval.scenario_id), []).append(
            (interval.start_at, interval.end_at)
        )
    for spans in by_scope.values():
        spans.sort()
        start, end = spans[0]
        for next_start, next_end in spans[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                total += end - start
                start, end = next_start, next_end
        total += end - start
    return total.total_seconds() / 86400


def _coverage(intervals: Sequence[EvaluationInterval]) -> dict[str, Any]:
    counts = {status: 0 for status in sorted(VALID_STATUSES)}
    by_type: dict[str, dict[str, Any]] = {}
    for interval in intervals:
        counts[interval.status] += 1
        type_counts = by_type.setdefault(
            interval.sensor_type, {status: 0 for status in sorted(VALID_STATUSES)}
        )
        type_counts[interval.status] += 1
    total = sum(counts.values())
    fractions = {key: _ratio(value, total) for key, value in counts.items()}
    for type_counts in by_type.values():
        type_total = sum(type_counts.values())
        type_counts["fractions"] = {
            key: _ratio(type_counts[key], type_total) for key in sorted(VALID_STATUSES)
        }
        type_counts["total"] = type_total
    return {"counts": counts, "fractions": fractions, "total": total, "by_sensor_type": by_type}


def evaluate_episodes(
    truth: Iterable[TruthEpisode | Mapping[str, Any] | object],
    episodes: Iterable[Episode],
    *,
    intervals: Iterable[EvaluationInterval | Mapping[str, Any] | object] = (),
    protocol_version: str = "stage1-eval-v1",
) -> EvaluationReport:
    """Greedily match warnings, scoring only explicitly observable exposure.

    When ``intervals`` is non-empty, candidates and truth inside ``unknown`` or
    ``exclude`` spans are omitted from precision/recall.  They remain visible as
    ignored IDs and in coverage.  Candidate time is ``confirmed_at``: using the
    back-dated episode start would violate the causal decision-time contract.
    """

    truths = sorted((coerce_truth(item) for item in truth), key=lambda item: item.start_at)
    eligibility = tuple(coerce_interval(item) for item in intervals)
    candidates = [episode for episode in episodes if episode.decision is Decision.CANDIDATE]
    scenarios = {episode.episode_id: warning_scenario_id(episode) for episode in candidates}

    ignored_warnings: list[str] = []
    scored_warnings: list[Episode] = []
    for warning in candidates:
        if eligibility:
            status = _status_at(
                warning.channel_id,
                scenarios[warning.episode_id],
                warning.confirmed_at,
                eligibility,
            )
            if status != "include":
                ignored_warnings.append(warning.episode_id)
                continue
        scored_warnings.append(warning)

    scored_truth: list[TruthEpisode] = []
    for item in truths:
        if eligibility:
            statuses = {
                _status_at(channel_id, item.scenario_id, item.start_at, eligibility)
                for channel_id in item.match_channel_ids
            }
            if "include" not in statuses:
                continue
        if not item.is_control:
            scored_truth.append(item)

    unused = {warning.episode_id for warning in scored_warnings}
    warnings_by_scope: dict[tuple[str, str], list[Episode]] = {}
    for warning in scored_warnings:
        scope = (warning.channel_id, scenarios[warning.episode_id])
        warnings_by_scope.setdefault(scope, []).append(warning)
    for warnings in warnings_by_scope.values():
        warnings.sort(key=lambda warning: (warning.confirmed_at, warning.episode_id))

    matches: list[Match] = []
    unmatched_truth: list[str] = []
    for item in scored_truth:
        if item.expected_cadence is None:
            raise ValueError(
                f"scored truth {item.episode_id!r} requires a confirmed expected cadence"
            )
        eligible_until = item.end_at + item.expected_cadence
        eligible_warnings = [
            warning
            for channel_id in item.match_channel_ids
            for warning in warnings_by_scope.get((channel_id, item.scenario_id), ())
            if warning.episode_id in unused
            and item.start_at <= warning.confirmed_at <= eligible_until
        ]
        chosen = min(
            eligible_warnings,
            key=lambda warning: (warning.confirmed_at, warning.episode_id),
            default=None,
        )
        if chosen is None:
            unmatched_truth.append(item.episode_id)
            continue
        unused.remove(chosen.episode_id)
        matches.append(
            Match(
                truth_episode_id=item.episode_id,
                warning_episode_id=chosen.episode_id,
                channel_id=chosen.channel_id,
                scenario_id=item.scenario_id,
                warning_at=chosen.confirmed_at,
                delay_seconds=(chosen.confirmed_at - item.start_at).total_seconds(),
            )
        )

    unmatched_warning_ids = tuple(
        warning.episode_id for warning in scored_warnings if warning.episode_id in unused
    )
    tp, fp, fn = len(matches), len(unmatched_warning_ids), len(unmatched_truth)
    precision, recall, f1 = _scores(tp, fp, fn)
    channel_days = _included_days(eligibility)
    warning_rate = None if not channel_days else fp / channel_days * 100

    all_scenarios = sorted(
        {item.scenario_id for item in truths}
        | set(scenarios.values())
        | {interval.scenario_id for interval in eligibility}
    )
    per_scenario: dict[str, Mapping[str, Any]] = {}
    truth_by_id = {item.episode_id: item for item in scored_truth}
    for scenario_id in all_scenarios:
        scenario_matches = [match for match in matches if match.scenario_id == scenario_id]
        scenario_fn = sum(
            truth_by_id[item_id].scenario_id == scenario_id for item_id in unmatched_truth
        )
        scenario_fp = sum(scenarios[item_id] == scenario_id for item_id in unmatched_warning_ids)
        scenario_tp = len(scenario_matches)
        p_value, r_value, f_value = _scores(scenario_tp, scenario_fp, scenario_fn)
        per_scenario[scenario_id] = {
            "true_positives": scenario_tp,
            "false_positives": scenario_fp,
            "false_negatives": scenario_fn,
            "precision": p_value,
            "recall": r_value,
            "f1": f_value,
        }

    delays = [match.delay_seconds for match in matches]
    return EvaluationReport(
        protocol_version=protocol_version,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        delay_seconds_median=_percentile(delays, 0.5),
        delay_seconds_p90=_percentile(delays, 0.9),
        included_channel_days=channel_days,
        false_warnings_per_100_channel_days=warning_rate,
        matches=tuple(matches),
        unmatched_truth_ids=tuple(unmatched_truth),
        unmatched_warning_ids=unmatched_warning_ids,
        ignored_warning_ids=tuple(ignored_warnings),
        coverage=_coverage(eligibility),
        per_scenario=per_scenario,
    )


def compare_variants(
    truth: Iterable[TruthEpisode | Mapping[str, Any] | object],
    runs: Mapping[str, Iterable[Episode]],
    *,
    intervals: Iterable[EvaluationInterval | Mapping[str, Any] | object] = (),
    baseline: str = "baseline",
) -> dict[str, Any]:
    """Evaluate threshold/ablation runs without interpreting a model as an improvement."""

    truth_items = tuple(truth)
    interval_items = tuple(intervals)
    reports = {
        name: evaluate_episodes(truth_items, episodes, intervals=interval_items)
        for name, episodes in runs.items()
    }
    if baseline not in reports:
        raise ValueError(f"baseline run {baseline!r} is missing")
    base = reports[baseline]
    comparison: dict[str, Any] = {}
    for name, report in reports.items():
        comparison[name] = {
            "report": report.to_record(),
            "delta_f1": None if report.f1 is None or base.f1 is None else report.f1 - base.f1,
            "delta_recall": (
                None
                if report.recall is None or base.recall is None
                else report.recall - base.recall
            ),
            "delta_false_warnings_per_100_channel_days": (
                None
                if report.false_warnings_per_100_channel_days is None
                or base.false_warnings_per_100_channel_days is None
                else report.false_warnings_per_100_channel_days
                - base.false_warnings_per_100_channel_days
            ),
        }
    isolation_name = next(
        (name for name in reports if name.lower().replace("-", "_") == "isolation_forest"),
        None,
    )
    isolation_forest = (
        {
            "status": "compared_same_protocol",
            "variant": isolation_name,
            "claim": "no automatic improvement claim; inspect reported deltas",
        }
        if isolation_name
        else {
            "status": "not_comparable",
            "reason": (
                "no pre-frozen IsolationForest warning run with identical truth, causal feature "
                "history, and eligibility exposure was supplied"
            ),
            "required_evidence": [
                "pre-frozen training period and feature schema",
                "candidate Episodes for the same scenario/channel intervals",
                "identical unknown/exclude decisions",
            ],
        }
    )
    expected = set(EXPECTED_SENSITIVITY_VARIANTS)
    supplied = set(reports)
    return {
        "baseline": baseline,
        "variants": comparison,
        "sensitivity_plan": {
            "expected": dict(EXPECTED_SENSITIVITY_VARIANTS),
            "supplied": sorted(supplied & expected),
            "missing": sorted(expected - supplied),
            "complete": expected <= supplied,
        },
        "method_comparisons": {"isolation_forest": isolation_forest},
    }
