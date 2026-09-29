"""Separate research admission ablations; never alter Q2/R1 or frozen R6.

Past contradictions remain excluded from features. A later exact, unambiguous
normal can be proposed as a new state checkpoint, NOT as proof of uptime or
of compatibility of the earlier messages. B must approve before model use.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

from stage1.features.hourly import HourlyConfig
from stage1.features.sparse_admission import SparseAdmissionStream
from stage1.shadow.stream import Observation
from stage1.state_labeling.operational import recent_normal_for_prediction, segment_at
from stage1.state_labeling.operational import source_is_full_archive
from stage1.state_labeling.rules import KNOWN_SENSOR_TYPES


VERSION = "q3-a-coverage-reentry-research-v1"
POLICIES = ("cold_start", "after_normal", "combined")
EVIDENCE_FIELDS = ("quality_rows_24h", "last_conflict_at", "last_hard_quality_at")
PAST_FIELDS = (
    "channel_id",
    "prediction_time",
    "sensor_type",
    "admission_status",
    "admission_reasons",
    "availability_status",
    "last_explicit_normal_at",
    "admission_evidence_through",
    "first_usable_at",
    "second_usable_at",
    "excluded_quality_count_24h",
    "ambiguous_seconds_24h",
    "blocking_qa_count_24h",
)


def propose(row: dict, evidence: dict, policy: str) -> dict:
    """Strict label-free interface on an independently certified Q2 snapshot."""
    if policy not in POLICIES or set(row) != set(PAST_FIELDS):
        raise ValueError("expected only certified past fields and a declared policy")
    if set(evidence) != set(EVIDENCE_FIELDS):
        raise ValueError("expected only raw past quality evidence")
    at = row["prediction_time"]
    if not isinstance(at, datetime) or at.tzinfo is not None or segment_at(at) is None:
        raise ValueError("prediction must be inside an accepted local archive segment")
    if (at.minute, at.second, at.microsecond) != (0, 0, 0):
        raise ValueError("prediction must be a whole hour")
    for name in (
        "last_explicit_normal_at",
        "admission_evidence_through",
        "first_usable_at",
        "second_usable_at",
        "last_conflict_at",
        "last_hard_quality_at",
    ):
        value = row[name] if name in row else evidence[name]
        if value is not None and (
            value.tzinfo is not None or value > at or segment_at(value) != segment_at(at)
        ):
            raise ValueError("future or cross-segment evidence")
    for value in (
        row["excluded_quality_count_24h"],
        row["ambiguous_seconds_24h"],
        row["blocking_qa_count_24h"],
        evidence["quality_rows_24h"],
    ):
        if type(value) is not int or value < 0:
            raise ValueError("invalid quality count")
    if evidence["quality_rows_24h"] != row["excluded_quality_count_24h"]:
        raise ValueError("raw quality count differs from certified snapshot")
    status, reasons = row["admission_status"], set(row["admission_reasons"])
    if (
        status not in {"eligible", "unknown", "excluded"}
        or ((status == "eligible") != (not reasons))
        or row["availability_status"] != "unknown"
    ):
        raise ValueError("inconsistent certified admission or availability")
    first, second = row["first_usable_at"], row["second_usable_at"]
    if second is not None and (first is None or second <= first):
        raise ValueError("second usable timestamp must strictly follow the first")
    normal = row["last_explicit_normal_at"]
    protected_ok = (
        status != "excluded"
        and row["sensor_type"] in KNOWN_SENSOR_TYPES
        and recent_normal_for_prediction(normal, at)
        and row["admission_evidence_through"] is not None
        and row["blocking_qa_count_24h"] == 0
        and row["ambiguous_seconds_24h"] == 0
    )
    relaxed = []
    if protected_ok and policy in {"cold_start", "combined"} and second is not None:
        if "insufficient_history" in reasons:
            reasons.remove("insufficient_history")
            relaxed.append("insufficient_history")
    conflict, hard = evidence["last_conflict_at"], evidence["last_hard_quality_at"]
    recovered = (
        protected_ok
        and evidence["quality_rows_24h"] > 0
        and conflict is not None
        and at - timedelta(hours=24) < conflict < normal
        and (hard is None or hard <= at - timedelta(hours=24))
    )
    if recovered and policy in {"after_normal", "combined"}:
        if "quality_exclusions_24h" in reasons:
            reasons.remove("quality_exclusions_24h")
            relaxed.append("quality_exclusions_24h")
    if not reasons and not protected_ok:
        raise ValueError("eligible snapshot is missing a protected guard")
    return {
        "research_status": "excluded"
        if status == "excluded"
        else "unknown"
        if reasons
        else "eligible",
        "research_reasons": sorted(reasons),
        "relaxed_reasons": sorted(relaxed),
    }


def policy_sql() -> str:
    """Input is `evidence`: certified causal fields plus raw ASOF quality fields."""
    known = ",".join("'" + k.replace("'", "''") + "'" for k in sorted(KNOWN_SENSOR_TYPES))
    protected = (
        "admission_status<>'excluded' AND last_explicit_normal_at IS NOT NULL "
        "AND last_explicit_normal_at>=prediction_time-INTERVAL '168 hours' "
        "AND admission_evidence_through IS NOT NULL AND blocking_qa_count_24h=0 "
        "AND ambiguous_seconds_24h=0 AND sensor_type IN (" + known + ")"
    )
    recovered = (
        "quality_rows_24h>0 AND last_conflict_at>prediction_time-INTERVAL '24 hours' "
        "AND last_conflict_at<last_explicit_normal_at AND "
        "(last_hard_quality_at IS NULL OR last_hard_quality_at<=prediction_time-INTERVAL '24 hours')"
    )
    outputs = []
    for policy in POLICIES:
        cold = "can_cold_start" if policy in {"cold_start", "combined"} else "false"
        recovery = "can_reenter" if policy in {"after_normal", "combined"} else "false"
        reasons = (
            "list_sort(list_filter(admission_reasons,r->NOT ((r='insufficient_history' AND "
            + cold
            + ") OR (r='quality_exclusions_24h' AND "
            + recovery
            + "))))"
        )
        outputs.extend(
            [
                f"{reasons} AS {policy}_reasons",
                "CASE WHEN admission_status='excluded' THEN 'excluded' WHEN len("
                + reasons
                + f")=0 THEN 'eligible' ELSE 'unknown' END AS {policy}_status",
            ]
        )
    return (
        "WITH guards AS (SELECT *,COALESCE("
        + protected
        + ",false) AS protected_ok FROM evidence), "
        "options AS (SELECT *,protected_ok AND second_usable_at IS NOT NULL AS can_cold_start,"
        "protected_ok AND COALESCE("
        + recovered
        + ",false) AS can_reenter FROM guards) SELECT *,"
        + ",".join(outputs)
        + " FROM options"
    )


class ResearchCoverageStream:
    """Independent raw replay adapter; consumes no labels or saved admission index."""

    def __init__(self):
        self.past = SparseAdmissionStream()
        self.state = {}

    def observe_records(self, records):
        group = [Observation.from_record(r) for r in records]
        self.past.observe_group(group)
        for row in group:
            event = row.event
            if not source_is_full_archive(row.source, event.timestamp):
                continue
            segment = segment_at(event.timestamp)
            item = self.state.get(event.channel_id)
            if item is None or item["segment"] != segment:
                item = {
                    "segment": segment,
                    "first": None,
                    "second": None,
                    "quality": deque(),
                    "last_conflict_at": None,
                    "last_hard_quality_at": None,
                }
                self.state[event.channel_id] = item
            flags = HourlyConfig().excluded_quality_flags.intersection(event.quality_flags)
            if flags:
                item["quality"].append(event.timestamp)
                field = (
                    "last_conflict_at"
                    if flags == {"channel_time_conflict"}
                    else "last_hard_quality_at"
                )
                item[field] = event.timestamp
            elif item["first"] is None:
                item["first"] = event.timestamp
            elif item["second"] is None and event.timestamp > item["first"]:
                item["second"] = event.timestamp
            while item["quality"] and item["quality"][0] <= event.timestamp - timedelta(hours=24):
                item["quality"].popleft()

    def evaluate(self, at, channels):
        originals = self.past.evaluate(at, channels)
        result = []
        for snapshot in originals:
            item = self.state.get(snapshot["channel_id"])
            if item is None or item["segment"] != segment_at(at):
                item = {
                    "first": None,
                    "second": None,
                    "quality": deque(),
                    "last_conflict_at": None,
                    "last_hard_quality_at": None,
                }
            while item["quality"] and item["quality"][0] <= at - timedelta(hours=24):
                item["quality"].popleft()
            past = {
                name: snapshot[name]
                for name in PAST_FIELDS
                if name
                not in {
                    "admission_evidence_through",
                    "first_usable_at",
                    "second_usable_at",
                    "excluded_quality_count_24h",
                    "ambiguous_seconds_24h",
                }
            }
            past.update(
                admission_evidence_through=snapshot["admission_through"],
                first_usable_at=item["first"],
                second_usable_at=item["second"],
                excluded_quality_count_24h=len(item["quality"]),
                ambiguous_seconds_24h=int(
                    "same_time_state_ambiguity" in snapshot["admission_reasons"]
                ),
            )
            evidence = {name: item[name] for name in EVIDENCE_FIELDS if name != "quality_rows_24h"}
            evidence["quality_rows_24h"] = len(item["quality"])
            result.append({**past, **evidence, **{p: propose(past, evidence, p) for p in POLICIES}})
        return result
