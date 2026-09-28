"""Restartable bounded M1 replay; atomically commit A state, B state and their outputs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import duckdb
import numpy as np
import psutil

from analysis.build_a2_hourly import _monthly_files
from analysis.r6_provenance import frozen_rule_sha256
from analysis.replay_shadow_pilot import COLUMNS, observation_groups, select_channels
from analysis.shadow_checkpoint_bundle import ReplaySession, load_bundle, save_bundle
from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.shadow_pilot import ShadowPolicy
from stage1.shadow.checkpoint import CheckpointError
from stage1.shadow.stream import PILOT_VERSION
from stage1.state_labeling.operational import ARCHIVE_SEGMENTS, segment_at


CODE_FILES = (
    "analysis/replay_shadow_checkpoint.py",
    "analysis/shadow_checkpoint_bundle.py",
    "analysis/replay_shadow_pilot.py",
    "stage1/shadow/checkpoint.py",
    "stage1/shadow/stream.py",
    "stage1/features/hourly.py",
    "stage1/features/r2.py",
    "stage1/state_labeling/rules.py",
    "stage1/state_labeling/operational.py",
    "stage1/state_labeling/registered_episodes.py",
    "ml/forecast/r6_rule.py",
    "ml/forecast/shadow_pilot.py",
)


def run(
    *,
    m1_dir: Path,
    output_dir: Path,
    freeze_path: Path,
    contract_path: Path,
    checkpoint_in: Path | None = None,
    stop_at: datetime | None = None,
) -> dict:
    started = time.perf_counter()
    policy = ShadowPolicy.from_freeze(freeze_path)
    contract = read_json(contract_path)
    if (
        contract["schema_version"] != PILOT_VERSION
        or contract["source_m1_manifest_sha256"] != sha256(m1_dir / "manifest.json")
        or contract["source_r6_freeze_lf_sha256"] != policy.freeze_sha256
        or read_json(m1_dir / "manifest.json")["status"] != "complete"
        or contract["automatic_actions_enabled"] is not False
    ):
        raise CheckpointError("archive replay lineage differs")
    settings = {
        "baseline_lookback_days": 28,
        "baseline_embargo_hours": 24,
        "maximum_feature_window_hours": 168,
        "event_retention_days": 37,
        "recent_explicit_normal_hours": 168,
    }
    if any(contract.get(key) != value for key, value in settings.items()):
        raise CheckpointError("archive admission policy differs")
    start = datetime.fromisoformat(contract["first_replay_start"])
    end = datetime.fromisoformat(contract["first_replay_end_exclusive"])
    segment = segment_at(start)
    if segment is None:
        raise CheckpointError("archive replay starts outside accepted segment")
    history_start = ARCHIVE_SEGMENTS[segment][0]
    files, missing = _monthly_files(m1_dir, history_start, end)
    if missing or not files:
        raise CheckpointError(f"archive source months absent: {missing}")
    maximum = contract["first_replay_max_channels"]
    if type(maximum) is not int or not 1 <= maximum <= 20:
        raise CheckpointError("bounded replay supports at most 20 channels")
    stop_at = stop_at or end
    if (
        stop_at.tzinfo is not None
        or (stop_at.minute, stop_at.second, stop_at.microsecond) != (0, 0, 0)
        or not start < stop_at <= end
    ):
        raise CheckpointError("stop_at must be a whole hour inside the pinned replay period")
    if output_dir.exists() or output_dir.with_name(output_dir.name + ".inprogress").exists():
        raise FileExistsError(output_dir)
    root = Path(__file__).resolve().parents[1]
    checkpoint_contract = root / "ml/shadow_checkpoint_contract_v1.json"
    if read_json(checkpoint_contract)["source_r6_freeze_lf_sha256"] != policy.freeze_sha256:
        raise CheckpointError("checkpoint contract belongs to another rule")
    identity = {
        "source_mode": "immutable_m1_archive_event_time_only_not_actual_arrival",
        "m1_manifest_sha256": sha256(m1_dir / "manifest.json"),
        "freeze_lf_sha256": policy.freeze_sha256,
        "contract_lf_sha256": frozen_rule_sha256(contract_path),
        "checkpoint_contract_lf_sha256": frozen_rule_sha256(checkpoint_contract),
        "files": [
            {"file": path.relative_to(m1_dir).as_posix(), "sha256": sha256(path)} for path in files
        ],
        "code_lf_sha256": {name: frozen_rule_sha256(root / name) for name in CODE_FILES},
    }
    session = None
    restore_seconds = 0.0
    if checkpoint_in is not None:
        before_restore = time.perf_counter()
        session = load_bundle(checkpoint_in, source_identity=identity, policy=policy)
        restore_seconds = time.perf_counter() - before_restore
        if session.start != start or session.end != end or session.next_prediction >= stop_at:
            raise CheckpointError("resume period differs or has no remaining prediction hours")
    latency = []
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        database.execute("SET memory_limit='2GB'")
        if session is None:
            selected = select_channels(database, files, history_start, start, maximum)
            session = ReplaySession(
                source_identity=identity,
                channels=[item["channel_id"] for item in selected],
                start=start,
                end=end,
                policy=policy,
            )
        database.execute(
            "CREATE TEMP TABLE selected_channels AS SELECT UNNEST(?::VARCHAR[]) AS channel_id",
            [session.channels],
        )
        lower, operator = (session.a.watermark, ">") if checkpoint_in else (history_start, ">=")
        reader = database.execute(
            "SELECT " + ",".join(COLUMNS) + " FROM read_parquet(?,hive_partitioning=false) e "
            "SEMI JOIN selected_channels USING(channel_id) WHERE timestamp " + operator + " ? "
            "AND timestamp<? AND split_part(replace(source,chr(92),'/'),'/',-1)="
            "'ext-journal-' || CAST(year(timestamp) AS VARCHAR) || '.7z' "
            "ORDER BY timestamp,channel_id,row_id",
            [[str(path) for path in files], lower, stop_at],
        ).to_arrow_reader(batch_size=25_000)
        groups = iter(observation_groups(row for batch in reader for row in batch.to_pylist()))
        group = next(groups, None)
        while group is not None and group[0].event.timestamp < session.next_prediction:
            session.observe_group(group)
            group = next(groups, None)
        while session.next_prediction < stop_at:
            before_hour = time.perf_counter()
            while group is not None and group[0].event.timestamp <= session.next_prediction:
                session.observe_group(group)
                group = next(groups, None)
            session.predict_hour()
            latency.append(time.perf_counter() - before_hour)
    memory = psutil.Process().memory_info()
    resources = {
        "elapsed_before_commit_seconds": time.perf_counter() - started,
        "restore_seconds": restore_seconds,
        "peak_working_set_before_commit_bytes": getattr(memory, "peak_wset", memory.rss),
        "hour_processing_p50_seconds": float(np.quantile(latency, 0.50)),
        "hour_processing_p95_seconds": float(np.quantile(latency, 0.95)),
        "hour_processing_p99_seconds": float(np.quantile(latency, 0.99)),
        "hour_latency_scope": "local processing only; excludes warmup, restore and bundle commit",
    }
    before_commit = time.perf_counter()
    manifest = save_bundle(session, output_dir, resources=resources)
    result = {
        "checkpoint_manifest": manifest,
        "resources": resources,
        "commit_seconds": time.perf_counter() - before_commit,
        "total_elapsed_seconds": time.perf_counter() - started,
        "peak_working_set_bytes": getattr(psutil.Process().memory_info(), "peak_wset", memory.rss),
        "resumed": checkpoint_in is not None,
    }
    print(
        f"committed {len(session.predictions)} rows through {session.a.watermark}; "
        f"resumed={checkpoint_in is not None}",
        flush=True,
    )
    return result


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, default=Path("ml/r6_frozen_rule_v1.json"))
    parser.add_argument("--contract", type=Path, default=Path("ml/shadow_pilot_contract_v1.json"))
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--stop-at", type=datetime.fromisoformat)
    args = parser.parse_args()
    result = run(
        m1_dir=args.m1_dir,
        output_dir=args.output_dir,
        freeze_path=args.freeze,
        contract_path=args.contract,
        checkpoint_in=args.checkpoint_in,
        stop_at=args.stop_at,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
