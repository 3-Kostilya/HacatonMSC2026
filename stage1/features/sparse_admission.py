"""Proposed missing-aware admission, not an approved population or model.

Reuse the accepted past-state tracker without changing frozen R6 inference.
Only three rich-statistics vetoes are relaxed. No scores or warnings are made.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Mapping, Sequence

from stage1.shadow.stream import FORBIDDEN_FIELDS, Observation, ShadowStream
from stage1.state_labeling.operational import recent_normal_for_prediction, segment_at
from stage1.state_labeling.operational import source_is_full_archive
from stage1.state_labeling.rules import KNOWN_SENSOR_TYPES
from stage1.value_quality import assess_value


VERSION = "q2-a-sparse-admission-proposal-v1"
RELAXED_REASONS = frozenset(
    {"baseline_unusable", "state_history_missing", "state_transitions_unavailable"}
)
BLOCKING_QA = frozenset(
    {"epoch_value_artifact", "temperature_service_code_candidate", "gas_above_physical_percent"}
)
FUTURE_FIELDS = FORBIDDEN_FIELDS | {
    "horizon_end",
    "prior_normal_at",
    "label_version",
    "future_label_reasons",
}


def candidate_from_past_snapshot(row: Mapping, *, blocking_qa_count_24h: int = 0) -> dict:
    """Relax absent statistics only; keep every other veto and excluded status.

    This function needs the causal ShadowStream snapshot, not A3 row_status
    alone: A3 does not by itself prove that no registered episode is active.
    """
    if (FUTURE_FIELDS - {"admission_status"}).intersection(row) or {
        "rule_score",
        "warning_emitted",
        "above_frozen_threshold",
    }.intersection(row):
        raise ValueError("expected unscored causal snapshot, not future labels or R6 decisions")
    status, original = row["admission_status"], set(row["admission_reasons"])
    if (
        status not in {"eligible", "unknown", "excluded"}
        or (status == "eligible" and original)
        or (status != "eligible" and not original)
        or type(blocking_qa_count_24h) is not int
        or blocking_qa_count_24h < 0
    ):
        raise ValueError("inconsistent past admission or QA count")
    at = row["prediction_time"]
    for name in ("history_through", "admission_through", "last_explicit_normal_at"):
        if row[name] is not None and row[name] > at:
            raise ValueError("snapshot contains future evidence")
    if row["availability_status"] != "unknown":
        raise ValueError("candidate cannot assert channel availability")
    remaining = original - RELAXED_REASONS
    if blocking_qa_count_24h:
        remaining.add("qa_unusable_measurement_24h")
    if (
        status != "excluded"
        and not remaining
        and (
            row["sensor_type"] not in KNOWN_SENSOR_TYPES
            or not recent_normal_for_prediction(row["last_explicit_normal_at"], at)
            or row["history_through"] is None
            or row["admission_through"] is None
        )
    ):
        raise ValueError("eligible snapshot lacks known type, recent normal or causal history")
    return {
        **row,
        "candidate_version": VERSION,
        "legacy_admission_status": status,
        "legacy_admission_reasons": sorted(original),
        "relaxed_data_reasons": sorted(original & RELAXED_REASONS),
        "blocking_qa_count_24h": blocking_qa_count_24h,
        "admission_status": (
            "excluded" if status == "excluded" else "unknown" if remaining else "eligible"
        ),
        "admission_reasons": sorted(remaining),
    }


class SparseAdmissionStream:
    """Research-only event-time admission; deliberately has no scoring API.

    The private snapshot dependency is pinned in the handoff manifest and
    covered by parity tests. It prevents a second, drifting R1 state machine.
    """

    def __init__(self) -> None:
        self._past = ShadowStream(threshold=7.1)
        self._qa: dict[str, tuple[int, deque[datetime]]] = {}

    def observe_records(self, records: Sequence[Mapping]) -> None:
        if any(FUTURE_FIELDS.intersection(record) for record in records):
            raise ValueError("future labels and retrospective admission are forbidden inputs")
        self.observe_group([Observation.from_record(record) for record in records])

    def observe_group(self, group: Sequence[Observation]) -> None:
        self._past.observe_group(group)
        for row in group:
            event = row.event
            if not source_is_full_archive(row.source, event.timestamp):
                continue
            segment = segment_at(event.timestamp)
            assert segment is not None
            previous = self._qa.get(event.channel_id)
            if previous is None or previous[0] != segment:
                self._qa[event.channel_id] = segment, deque()
            queue = self._qa[event.channel_id][1]
            quality = assess_value(event.sensor_type, event.value_state or "", event.value_numeric)
            if quality.category in BLOCKING_QA:
                queue.append(event.timestamp)
            self._prune_qa(queue, event.timestamp)

    @staticmethod
    def _prune_qa(queue: deque[datetime], at: datetime) -> None:
        while queue and queue[0] <= at - timedelta(hours=24):
            queue.popleft()

    def evaluate(self, at: datetime, channels: Sequence[str]) -> list[dict]:
        """Close a watermark; return one decision for every requested channel."""
        if at.tzinfo is not None or (at.minute, at.second, at.microsecond) != (0, 0, 0):
            raise ValueError("prediction time must be a naive local whole hour")
        if self._past.watermark is not None and at <= self._past.watermark:
            raise ValueError("prediction watermarks must be strictly increasing")
        if self._past._last_group is not None and self._past._last_group[0] > at:
            raise ValueError("future observations already consumed before prediction")
        if (
            not channels
            or len(set(channels)) != len(channels)
            or any(not isinstance(channel, str) or not channel.strip() for channel in channels)
        ):
            raise ValueError("configured channels must be nonempty and unique")
        results = []
        for channel in channels:
            qa_count = 0
            previous = self._qa.get(channel)
            if previous is not None and previous[0] == segment_at(at):
                self._prune_qa(previous[1], at)
                qa_count = len(previous[1])
            results.append(
                candidate_from_past_snapshot(
                    self._past._snapshot(channel, at), blocking_qa_count_24h=qa_count
                )
            )
        self._past.watermark = at
        return results

    @property
    def accepted_rows(self) -> int:
        return self._past.accepted_rows

    @property
    def retained_event_count(self) -> int:
        return self._past.retained_event_count
