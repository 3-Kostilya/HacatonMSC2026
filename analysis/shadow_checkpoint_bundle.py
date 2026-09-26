"""Atomic archive-replay commit: A history, B cooldown, source cursor and all published rows."""

from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.replay_shadow_pilot import B_INPUT_SCHEMA, OUTPUT_SCHEMA, to_b_input
from analysis.run_shadow_pilot_b import DECISION_SCHEMA
from analysis.train_r4_discrete_baselines import sha256
from ml.forecast.shadow_pilot import ShadowPolicy, ShadowState, run_shadow_batch
from stage1.shadow.checkpoint import (
    CheckpointError,
    canonical,
    digest,
    parse_json,
    restore,
    snapshot,
)
from stage1.shadow.stream import ShadowStream
from stage1.state_labeling.operational import segment_at


BUNDLE_VERSION = "shadow-a-b-atomic-replay-checkpoint-v1"
HOUR = timedelta(hours=1)
FILES = (
    "checkpoint.json",
    "shadow_predictions.parquet",
    "hours.parquet",
    "shadow_decisions.parquet",
    "report.json",
)


def compare_decisions(a_rows: list[dict], b_rows: list[dict]) -> None:
    if len(a_rows) != len(b_rows):
        raise CheckpointError("A/B decision counts differ")
    for a, b in zip(a_rows, b_rows, strict=True):
        if (
            a["channel_id"] != b["channel_id"]
            or a["prediction_time"].isoformat() != b["prediction_time"]
            or a["rule_score"] != b["rule_score"]
            or a["above_frozen_threshold"] != b["threshold_crossed"]
            or a["warning_emitted"] != b["shadow_warning"]
            or a["admission_status"] != b["admission_status"]
            or b["delivery_mode"] != "record_only"
            or b["automatic_action_taken"]
        ):
            raise CheckpointError("A/B decision or record-only contract differs")


class ReplaySession:
    """Small bounded diagnostic session, including its committed output prefix."""

    def __init__(
        self,
        *,
        source_identity: dict,
        channels: list[str],
        start: datetime,
        end: datetime,
        policy: ShadowPolicy,
    ) -> None:
        if (
            not source_identity
            or not channels
            or len(set(channels)) != len(channels)
            or len(channels) > 20
            or any(not isinstance(c, str) or not c for c in channels)
            or end <= start
            or end - start > timedelta(days=31)
            or any(
                at.tzinfo is not None or (at.minute, at.second, at.microsecond) != (0, 0, 0)
                for at in (start, end)
            )
            or segment_at(start) is None
            or segment_at(start) != segment_at(end - HOUR)
        ):
            raise CheckpointError("invalid bounded archive session contract")
        self.source_identity, self.channels = source_identity, list(channels)
        self.start, self.end, self.next_prediction = start, end, start
        self.policy = policy
        self.a = ShadowStream(threshold=policy.threshold)
        self.b = ShadowState()
        self.predictions: list[dict] = []
        self.decisions: list[dict] = []

    def observe_group(self, group) -> None:
        if any(row.event.channel_id not in self.channels for row in group):
            raise CheckpointError("source group is outside the pinned channel roster")
        self.a.observe_group(group)

    def predict_hour(self) -> list[dict]:
        if self.next_prediction >= self.end:
            raise CheckpointError("session already complete")
        predictions = self.a.predict(self.next_prediction, self.channels)
        decisions = run_shadow_batch([to_b_input(row) for row in predictions], self.b, self.policy)
        compare_decisions(predictions, decisions)
        self.predictions.extend(predictions)
        self.decisions.extend(decisions)
        self.next_prediction += HOUR
        return predictions

    def checkpoint(self) -> dict:
        if (
            self.a.watermark is None
            or self.a.watermark != self.next_prediction - HOUR
            or self.b.last_prediction_at != dict.fromkeys(self.channels, self.a.watermark)
        ):
            raise CheckpointError("A/B checkpoint must follow a complete roster hour")
        return {
            "schema_version": BUNDLE_VERSION,
            "source_identity": self.source_identity,
            "channels": self.channels,
            "start": self.start.isoformat(),
            "end_exclusive": self.end.isoformat(),
            "next_prediction": self.next_prediction.isoformat(),
            "source_cursor": {
                "closed_through": self.a.watermark.isoformat(),
                "last_complete_group": snapshot(self.a)["last_group"],
                "accepted_rows": self.a.accepted_rows,
                "ignored_source_rows": self.a.ignored_source_rows,
                "resume_policy": "pinned_archive_query_timestamp_strictly_after_closed_through",
            },
            "a_state": snapshot(self.a),
            "b_state": self.b.checkpoint(self.policy),
        }


def restore_session(
    payload: dict,
    *,
    source_identity: dict,
    policy: ShadowPolicy,
    predictions: list[dict],
    decisions: list[dict],
) -> ReplaySession:
    try:
        if (
            payload["schema_version"] != BUNDLE_VERSION
            or payload["source_identity"] != source_identity
        ):
            raise CheckpointError("checkpoint source, code or contract fingerprint differs")
        session = ReplaySession(
            source_identity=source_identity,
            channels=payload["channels"],
            start=datetime.fromisoformat(payload["start"]),
            end=datetime.fromisoformat(payload["end_exclusive"]),
            policy=policy,
        )
        session.a = restore(payload["a_state"])
        session.b = ShadowState.restore(payload["b_state"], policy)
        session.next_prediction = datetime.fromisoformat(payload["next_prediction"])
        if (
            not session.start < session.next_prediction <= session.end
            or session.next_prediction != session.a.watermark + HOUR
            or set(session.a.channels) - set(session.channels)
        ):
            raise CheckpointError("checkpoint clock or channel roster differs")
        expected = int((session.next_prediction - session.start) / HOUR) * len(session.channels)
        if len(predictions) != expected or len(decisions) != expected:
            raise CheckpointError("checkpoint committed output prefix is incomplete")
        clock = session.start
        replayed_b = ShadowState()
        for offset in range(0, expected, len(session.channels)):
            hour = predictions[offset : offset + len(session.channels)]
            if [row["channel_id"] for row in hour] != session.channels or any(
                row["prediction_time"] != clock for row in hour
            ):
                raise CheckpointError("checkpoint output prefix has duplicate or missing keys")
            recomputed = run_shadow_batch([to_b_input(row) for row in hour], replayed_b, policy)
            actual = decisions[offset : offset + len(session.channels)]
            compare_decisions(hour, actual)
            if actual != recomputed:
                raise CheckpointError("checkpoint B decisions differ from frozen replay")
            clock += HOUR
        if replayed_b.checkpoint(policy) != session.b.checkpoint(policy):
            raise CheckpointError("checkpoint B cooldown differs from committed warnings")
        a_warnings = {
            channel: state.last_warning_at
            for channel, state in session.a.channels.items()
            if state.last_warning_at is not None
        }
        if a_warnings != session.b.last_warning_at:
            raise CheckpointError("checkpoint A/B warning state differs")
        session.predictions, session.decisions = predictions, decisions
        if session.checkpoint() != payload:
            raise CheckpointError("checkpoint cursor, versions or extra fields differ")
        return session
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError(f"invalid replay checkpoint: {error}") from error


def save_bundle(session: ReplaySession, directory: Path, *, resources: dict | None = None) -> dict:
    """Immutable directory publication; no state can commit ahead of its outputs.

    Files are flushed before the same-filesystem rename. This covers process
    interruption, not an OS/filesystem guarantee against sudden power loss.
    """
    pending = directory.with_name(directory.name + ".inprogress")
    if directory.exists() or pending.exists():
        raise FileExistsError(f"checkpoint destination already exists: {directory}")
    payload = session.checkpoint()
    restore_session(
        payload,
        source_identity=session.source_identity,
        policy=session.policy,
        predictions=session.predictions,
        decisions=session.decisions,
    )
    pending.mkdir(parents=True)
    (pending / "checkpoint.json").write_bytes(canonical(payload) + b"\n")
    for name, rows, schema in (
        ("shadow_predictions.parquet", session.predictions, OUTPUT_SCHEMA),
        ("hours.parquet", [to_b_input(row) for row in session.predictions], B_INPUT_SCHEMA),
        ("shadow_decisions.parquet", session.decisions, DECISION_SCHEMA),
    ):
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema), pending / name, compression="zstd"
        )
    report = {
        "schema_version": BUNDLE_VERSION,
        "purpose": "historical_process_restart_check_not_new_model_quality",
        "configured_channel_hours_in_period": int((session.end - session.start) / HOUR)
        * len(session.channels),
        "committed_channel_hours": len(session.predictions),
        "conditionally_scored_hours": sum(
            row["rule_score"] is not None for row in session.predictions
        ),
        "recorded_shadow_warnings": sum(row["shadow_warning"] for row in session.decisions),
        "resources_this_process": resources,
        "deployment_approved": False,
        "live_source_supported": False,
        "joint_pilot_acceptance": False,
        "quality_metrics_computed": False,
        "automatic_actions_enabled": False,
    }
    (pending / "report.json").write_bytes(canonical(report) + b"\n")
    for name in FILES:
        with (pending / name).open("r+b") as handle:
            os.fsync(handle.fileno())
    manifest = {
        "schema_version": BUNDLE_VERSION,
        "status": "closed_hour_checkpoint_record_only",
        "source_identity_sha256": digest(session.source_identity),
        "closed_through": session.a.watermark.isoformat(),
        "committed_rows": len(session.predictions),
        "files_sha256": {name: sha256(pending / name) for name in FILES},
    }
    with (pending / "manifest.json").open("wb") as handle:
        handle.write(canonical(manifest) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    pending.rename(directory)
    return manifest


def load_bundle(directory: Path, *, source_identity: dict, policy: ShadowPolicy) -> ReplaySession:
    try:
        if directory.name.endswith(".inprogress"):
            raise CheckpointError("an unpublished checkpoint cannot be restored")
        manifest = parse_json((directory / "manifest.json").read_bytes())
        if (
            manifest["schema_version"] != BUNDLE_VERSION
            or manifest["status"] != "closed_hour_checkpoint_record_only"
            or manifest["source_identity_sha256"] != digest(source_identity)
            or set(manifest["files_sha256"]) != set(FILES)
        ):
            raise CheckpointError("checkpoint manifest lineage or files differ")
        for name in FILES:
            if manifest["files_sha256"][name] != sha256(directory / name):
                raise CheckpointError(f"checkpoint file integrity failed: {name}")
        payload = parse_json((directory / "checkpoint.json").read_bytes())
        predictions = pq.ParquetFile(directory / "shadow_predictions.parquet").read().to_pylist()
        decisions = pq.ParquetFile(directory / "shadow_decisions.parquet").read().to_pylist()
        inputs = pq.ParquetFile(directory / "hours.parquet").read().to_pylist()
        if inputs != [to_b_input(row) for row in predictions]:
            raise CheckpointError("checkpoint B input differs from A output")
        session = restore_session(
            payload,
            source_identity=source_identity,
            policy=policy,
            predictions=predictions,
            decisions=decisions,
        )
        if (
            manifest["committed_rows"] != len(predictions)
            or manifest["closed_through"] != session.a.watermark.isoformat()
        ):
            raise CheckpointError("checkpoint publication cursor differs")
        return session
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError(f"checkpoint unavailable or damaged: {error}") from error
