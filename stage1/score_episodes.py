"""Causal B3 state machine: anomaly scores -> episodes -> notifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import math
from typing import Iterable, Mapping

from stage1.contracts import Decision, Episode, Origin


SCORE_STATUSES = frozenset({"eligible", "unknown", "excluded"})


def _local_time(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is not None:
        raise ValueError(f"{name} must use the journal's local naive time")


@dataclass(frozen=True, slots=True)
class ScoreObservation:
    channel_id: str
    as_of: datetime
    anomaly_type: str
    score: float | None
    status: str
    sensor_type: str
    sensor_group: str
    method: str
    method_version: str
    status_reasons: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    cause_hypothesis: str = "unknown"
    object_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "channel_id",
            "anomaly_type",
            "sensor_type",
            "sensor_group",
            "method",
            "method_version",
            "cause_hypothesis",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        _local_time("as_of", self.as_of)
        if self.status not in SCORE_STATUSES:
            raise ValueError("invalid score status")
        if self.status == "eligible":
            if (
                self.score is None
                or isinstance(self.score, bool)
                or not math.isfinite(self.score)
                or not 0 <= self.score <= 1
            ):
                raise ValueError("eligible score must be finite and between 0 and 1")
        elif self.score is not None or not self.status_reasons:
            raise ValueError("unavailable score must be null and have a reason")

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.channel_id,
            self.sensor_type,
            self.sensor_group,
            self.anomaly_type,
            self.cause_hypothesis,
        )


def score_observation_from_record(record: Mapping[str, object]) -> ScoreObservation:
    """Adapt the M0 anomaly-score envelope plus optional A3 semantic columns."""

    moment = record["as_of"]
    if isinstance(moment, str):
        moment = datetime.fromisoformat(moment)
    if not isinstance(moment, datetime):
        raise ValueError("as_of must be a datetime or ISO timestamp")
    method = str(record["method"])
    return ScoreObservation(
        channel_id=str(record["channel_id"]),
        as_of=moment,
        anomaly_type=str(record.get("anomaly_type") or method),
        score=(
            record.get("score")
            if isinstance(record.get("score"), (int, float))
            and not isinstance(record.get("score"), bool)
            else None
        ),
        status=str(record.get("score_status", record.get("status", "unknown"))),
        status_reasons=tuple(record.get("score_reasons", record.get("status_reasons", ()))),
        sensor_type=str(record.get("sensor_type") or "unknown"),
        sensor_group=str(record.get("sensor_group") or "unknown"),
        method=method,
        method_version=str(record["method_version"]),
        evidence=tuple(record.get("evidence", ())),
        cause_hypothesis=str(record.get("cause_hypothesis") or "unknown"),
        object_id=str(record["object_id"]) if record.get("object_id") is not None else None,
    )


@dataclass(frozen=True, slots=True)
class ScoreEpisodeConfig:
    enter_threshold: float = 0.8
    recovery_threshold: float = 0.4
    confirmation_observations: int = 3
    recovery_observations: int = 3
    maximum_confirmation_gap: timedelta | None = timedelta(hours=2)
    ruleset_version: str = "score-episodes-v1"

    def __post_init__(self) -> None:
        if not 0 <= self.recovery_threshold < self.enter_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= recovery < enter <= 1")
        if self.confirmation_observations < 1 or self.recovery_observations < 1:
            raise ValueError("confirmation counts must be positive")
        if self.maximum_confirmation_gap is not None and self.maximum_confirmation_gap <= timedelta(
            0
        ):
            raise ValueError("maximum_confirmation_gap must be positive")
        if not self.ruleset_version:
            raise ValueError("ruleset_version must be non-empty")


@dataclass(slots=True)
class _ActiveEpisode:
    start_at: datetime
    confirmed_at: datetime
    peak_score: float
    peak_at: datetime
    last_supported_at: datetime
    evidence: list[str]
    methods: set[str]
    method_versions: set[str]
    observation_count: int
    unknown_reasons: set[str] = field(default_factory=set)
    recovery_run: list[ScoreObservation] = field(default_factory=list)


def _episode_id(key: tuple[str, ...], start_at: datetime, ruleset_version: str) -> str:
    payload = "|".join((*key, start_at.isoformat(), ruleset_version)).encode("utf-8")
    return "score-episode-" + hashlib.sha256(payload).hexdigest()[:20]


def _to_episode(
    key: tuple[str, str, str, str, str],
    active: _ActiveEpisode,
    config: ScoreEpisodeConfig,
    *,
    end_at: datetime | None,
    recovery_started_at: datetime | None,
    object_id: str | None,
) -> Episode:
    channel_id, sensor_type, sensor_group, anomaly_type, cause = key
    metadata = {
        "peak_at": active.peak_at.isoformat(sep=" "),
        "last_confirmed_at": active.last_supported_at.isoformat(sep=" "),
        "score_observation_count": active.observation_count,
        "methods": sorted(active.methods),
        "method_versions": sorted(active.method_versions),
        "unknown_reasons_during_episode": sorted(active.unknown_reasons),
    }
    if recovery_started_at is not None:
        metadata["recovery_started_at"] = recovery_started_at.isoformat(sep=" ")
    evidence = tuple(dict.fromkeys(active.evidence)) or ("score_threshold_confirmation",)
    if end_at is not None:
        evidence = tuple(dict.fromkeys((*evidence, "sustained_recovery_confirmed")))
    return Episode(
        episode_id=_episode_id(key, active.start_at, config.ruleset_version),
        channel_id=channel_id,
        sensor_type=sensor_type,
        sensor_group=sensor_group,
        anomaly_type=anomaly_type,
        decision=Decision.CANDIDATE,
        start_at=active.start_at,
        confirmed_at=active.confirmed_at,
        end_at=end_at,
        ruleset_version=config.ruleset_version,
        evidence=evidence,
        observation_quality=tuple(sorted(active.unknown_reasons)),
        cause_hypothesis=cause,
        object_id=object_id,
        score=active.peak_score,
        origin=Origin.OBSERVED,
        metadata=metadata,
    )


def build_score_episodes(
    observations: Iterable[ScoreObservation],
    config: ScoreEpisodeConfig | None = None,
) -> list[Episode]:
    """Build episodes independently per channel/kind using only each causal prefix."""

    cfg = config or ScoreEpisodeConfig()
    grouped: dict[tuple[str, str, str, str, str], list[ScoreObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.key, []).append(observation)
    emitted: list[Episode] = []
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda item: item.as_of)
        if len({item.as_of for item in rows}) != len(rows):
            raise ValueError(f"duplicate score timestamp for {key!r}")
        object_ids = {item.object_id for item in rows if item.object_id is not None}
        if len(object_ids) > 1:
            raise ValueError(f"multiple object IDs for {key!r}")
        object_id = next(iter(object_ids), None)
        high_run: list[ScoreObservation] = []
        active: _ActiveEpisode | None = None
        previous_at: datetime | None = None
        for row in rows:
            gap_breaks_run = (
                previous_at is not None
                and cfg.maximum_confirmation_gap is not None
                and row.as_of - previous_at > cfg.maximum_confirmation_gap
            )
            previous_at = row.as_of
            if row.status != "eligible":
                high_run.clear()
                if active is not None:
                    active.recovery_run.clear()
                    active.unknown_reasons.update(row.status_reasons)
                continue
            assert row.score is not None
            if active is None:
                if gap_breaks_run:
                    high_run.clear()
                if row.score >= cfg.enter_threshold:
                    high_run.append(row)
                    if len(high_run) >= cfg.confirmation_observations:
                        first = high_run[-cfg.confirmation_observations]
                        peak = max(high_run, key=lambda item: item.score or 0)
                        active = _ActiveEpisode(
                            start_at=first.as_of,
                            confirmed_at=row.as_of,
                            peak_score=peak.score or 0,
                            peak_at=peak.as_of,
                            last_supported_at=row.as_of,
                            evidence=[value for item in high_run for value in item.evidence],
                            methods={item.method for item in high_run},
                            method_versions={item.method_version for item in high_run},
                            observation_count=len(high_run),
                        )
                        high_run.clear()
                else:
                    high_run.clear()
                continue

            active.observation_count += 1
            active.methods.add(row.method)
            active.method_versions.add(row.method_version)
            active.evidence.extend(row.evidence)
            if row.score > active.peak_score:
                active.peak_score = row.score
                active.peak_at = row.as_of
            if row.score > cfg.recovery_threshold:
                active.last_supported_at = row.as_of
            if gap_breaks_run:
                active.recovery_run.clear()
            if row.score <= cfg.recovery_threshold:
                active.recovery_run.append(row)
                if len(active.recovery_run) >= cfg.recovery_observations:
                    recovery_started_at = active.recovery_run[-cfg.recovery_observations].as_of
                    emitted.append(
                        _to_episode(
                            key,
                            active,
                            cfg,
                            end_at=row.as_of,
                            recovery_started_at=recovery_started_at,
                            object_id=object_id,
                        )
                    )
                    active = None
                    high_run.clear()
            else:
                active.recovery_run.clear()
        if active is not None:
            emitted.append(
                _to_episode(
                    key,
                    active,
                    cfg,
                    end_at=None,
                    recovery_started_at=None,
                    object_id=object_id,
                )
            )
    return sorted(emitted, key=lambda item: (item.start_at, item.channel_id, item.anomaly_type))


@dataclass(frozen=True, slots=True)
class EpisodeNotification:
    notification_id: str
    episode_id: str
    channel_id: str
    emitted_at: datetime
    anomaly_type: str


@dataclass(frozen=True, slots=True)
class NotificationResult:
    emitted: tuple[EpisodeNotification, ...]
    suppressed_duplicate_episode_ids: tuple[str, ...]


def build_notifications(
    episodes: Iterable[Episode], *, cooldown: timedelta = timedelta(hours=1)
) -> NotificationResult:
    """Emit once per episode; cooldown suppresses replay, never a recovered new episode."""

    if cooldown < timedelta(0):
        raise ValueError("cooldown cannot be negative")
    emitted = []
    seen: dict[str, datetime] = {}
    suppressed = []
    for episode in sorted(episodes, key=lambda item: (item.confirmed_at, item.episode_id)):
        previous = seen.get(episode.episode_id)
        if previous is not None and episode.confirmed_at <= previous + cooldown:
            suppressed.append(episode.episode_id)
            continue
        notification_id = (
            "notification-"
            + hashlib.sha256(
                f"{episode.episode_id}|{episode.confirmed_at.isoformat()}".encode("utf-8")
            ).hexdigest()[:20]
        )
        emitted.append(
            EpisodeNotification(
                notification_id,
                episode.episode_id,
                episode.channel_id,
                episode.confirmed_at,
                episode.anomaly_type,
            )
        )
        seen[episode.episode_id] = episode.confirmed_at
    return NotificationResult(tuple(emitted), tuple(suppressed))


@dataclass(frozen=True, slots=True)
class KindTruth:
    truth_id: str
    channel_id: str
    start_at: datetime
    end_at: datetime
    anomaly_type: str


def evaluate_anomaly_kinds(
    truth: Iterable[KindTruth], episodes: Iterable[Episode]
) -> dict[str, object]:
    """Score impact detection and anomaly-kind classification separately."""

    expected = sorted(truth, key=lambda item: (item.start_at, item.truth_id))
    candidates = [episode for episode in episodes if episode.decision is Decision.CANDIDATE]
    unused = set(range(len(candidates)))
    matches = []
    for item in expected:
        eligible = [
            index
            for index in unused
            if candidates[index].channel_id == item.channel_id
            and item.start_at <= candidates[index].confirmed_at <= item.end_at
        ]
        if not eligible:
            matches.append({"truth_id": item.truth_id, "episode_id": None, "kind_correct": False})
            continue
        index = min(eligible, key=lambda value: candidates[value].confirmed_at)
        unused.remove(index)
        episode = candidates[index]
        matches.append(
            {
                "truth_id": item.truth_id,
                "episode_id": episode.episode_id,
                "expected_kind": item.anomaly_type,
                "actual_kind": episode.anomaly_type,
                "kind_correct": episode.anomaly_type == item.anomaly_type,
            }
        )
    detected = sum(item["episode_id"] is not None for item in matches)
    correct = sum(item["kind_correct"] for item in matches)
    return {
        "truth_episodes": len(expected),
        "detected_impacts": detected,
        "correct_kinds": correct,
        "impact_recall": detected / len(expected) if expected else None,
        "kind_accuracy_on_detected": correct / detected if detected else None,
        "unmatched_warning_count": len(unused),
        "matches": matches,
    }
