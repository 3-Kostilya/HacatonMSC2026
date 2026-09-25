"""Independently recompute sampled full R3 A rows at their prediction time."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys

import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_a2_hourly import _monthly_files  # noqa: E402
from analysis.build_r3_full_month import CONTEXT, _channel_events  # noqa: E402
from analysis.r2_b2_handoff import load_b2_for_a2  # noqa: E402
from stage1.features.hourly import FeatureEvent, feature_at  # noqa: E402
from stage1.features.r2 import build_state_history_rows  # noqa: E402
from stage1.features.r3 import (  # noqa: E402
    A2_FEATURE_FIELDS,
    R2_DIAGNOSTIC_FIELDS,
    R2_FEATURE_FIELDS,
)
from stage1.state_labeling.operational import ARCHIVE_SEGMENTS, segment_at  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(*, m1_manifest: Path, b2_dir: Path, month_dir: Path,
          sample_size: int = 50) -> dict:
    m1_manifest = m1_manifest.resolve()
    b2_dir = b2_dir.resolve()
    month_dir = month_dir.resolve()
    manifest_path = month_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete_month" or (
        manifest.get("source_m1_manifest_sha256") != _sha256(m1_manifest)
    ) or manifest.get("source_b2_catalog_manifest_sha256") != _sha256(
        b2_dir / "manifest.json"
    ):
        raise ValueError("full R3 month has wrong publication or source lineage")
    feature_path = month_dir / "features.parquet"
    status_path = month_dir / "row_status.parquet"
    for path in (feature_path, status_path):
        if _sha256(path) != manifest["files"][path.name]["sha256"]:
            raise ValueError(f"R3 month Parquet SHA-256 mismatch: {path}")
    if not 1 <= sample_size <= 500:
        raise ValueError("sample_size must be 1-500")
    database = duckdb.connect(":memory:")
    try:
        samples = database.execute(
            """SELECT f.*, s.* EXCLUDE (channel_id, prediction_time)
               FROM read_parquet(?) AS f
               JOIN read_parquet(?) AS s USING (channel_id, prediction_time)
               WHERE hash(f.channel_id, f.prediction_time) % 10000 = 0
               ORDER BY hash(f.channel_id, f.prediction_time)
               LIMIT ?""",
            [str(feature_path), str(status_path), sample_size],
        ).to_arrow_table().to_pylist()
    finally:
        database.close()
    if not samples:
        raise ValueError("deterministic audit sample is empty")
    start, end = datetime.fromisoformat(manifest["start_at"]), datetime.fromisoformat(
        manifest["end_at"]
    )
    segment = segment_at(start)
    assert segment is not None
    context_start = max(start - CONTEXT, ARCHIVE_SEGMENTS[segment][0])
    files, missing = _monthly_files(m1_manifest.parent, context_start, end)
    if missing:
        raise ValueError(f"sample source months missing: {missing}")
    channels = sorted({row["channel_id"] for row in samples})
    events = dict(_channel_events(files, channels, context_start, end))
    catalog = load_b2_for_a2(b2_dir, local_m1_manifest=m1_manifest, channels=channels)
    episodes: dict[str, list] = defaultdict(list)
    for episode in catalog.episodes:
        episodes[episode.channel_id].append(episode)
    failures = []
    for row in samples:
        channel, t = row["channel_id"], row["prediction_time"]
        history = events.get(channel, [])
        if not history:
            failures.append(f"{channel}@{t}: no source events")
            continue
        fit_end = t.replace(hour=0) - timedelta(hours=168 + 24)
        expected = feature_at(history, channel, t, baseline_fit_end_at=fit_end)
        for name in A2_FEATURE_FIELDS:
            if row[name] != expected[name]:
                failures.append(f"{channel}@{t}: A2 {name}")
        r2 = build_state_history_rows(
            [{**expected, "run_id": "asof-audit"}], history,
            source_a2_manifest_sha256="a" * 64,
            completed_episodes=episodes.get(channel, []),
        ).to_pylist()[0]
        for name in (*R2_FEATURE_FIELDS, *R2_DIAGNOSTIC_FIELDS):
            if row[name] != r2[name]:
                failures.append(f"{channel}@{t}: R2 {name}")
        future = FeatureEvent(
            channel, t + timedelta(minutes=1), True,
            value_state="Неисправен", sensor_type=row["sensor_type"],
        )
        with_future = feature_at([*history, future], channel, t, baseline_fit_end_at=fit_end)
        if any(with_future[name] != expected[name] for name in A2_FEATURE_FIELDS):
            failures.append(f"{channel}@{t}: future event changed past features")
    return {
        "month": manifest["month"],
        "source_month_manifest_sha256": _sha256(manifest_path),
        "sampled_rows": len(samples),
        "sampled_channels": len(channels),
        "sampled_completed_episode_rows": sum(
            row["last_completed_episode_end_age_seconds"] is not None
            for row in samples
        ),
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-manifest", required=True, type=Path)
    parser.add_argument("--b2-dir", required=True, type=Path)
    parser.add_argument("--month-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(
        m1_manifest=args.m1_manifest, b2_dir=args.b2_dir,
        month_dir=args.month_dir, sample_size=args.sample_size,
    )
    if args.output:
        if args.output.exists():
            raise FileExistsError("R3 A as-of audit output exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
