"""Independently bound Q2 episode recall from B's saved positive prediction rows.

This is a retrospective oracle with future labels. It is never a predictor.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


COOLDOWN = timedelta(hours=24)
POSITIVE_COLUMNS = (
    "channel_id", "prediction_time", "sensor_type", "target_episode_id",
    "label_available_at",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def channel_capacity(times: list[pd.Timestamp], cooldown: timedelta = COOLDOWN) -> tuple[int, int]:
    """Return independent backward-DP and forward-greedy capacities."""
    if cooldown <= timedelta(0) or times != sorted(times):
        raise ValueError("positive times must be sorted and cooldown positive")
    best = [0] * (len(times) + 1)
    for index in range(len(times) - 1, -1, -1):
        next_compatible = bisect_left(times, times[index] + cooldown, index + 1)
        best[index] = max(best[index + 1], 1 + best[next_compatible])
    greedy = 0
    last: pd.Timestamp | None = None
    for at in times:
        if last is None or at - last >= cooldown:
            greedy += 1
            last = at
    return best[0], greedy


def verify(*, experiment: Path, q2_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    experiment_manifest = _read_json(experiment / "manifest.json")
    experiment_report = _read_json(experiment / "report.json")
    q2_manifest = _read_json(q2_dir / "manifest.json")
    if (experiment_manifest["schema_version"] != "q2-b-expanded-validation-v1"
            or experiment_manifest["report_sha256"] != _sha256(experiment / "report.json")
            or experiment_report["q2_manifest_sha256"] != _sha256(q2_dir / "manifest.json")):
        raise ValueError("saved B predictions do not match Q2 lineage")
    scores = []
    for item in experiment_manifest["score_files"]:
        path = experiment / item["name"]
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"score file changed: {path}")
        scores.append(str(path))
    if len(scores) != 12:
        raise ValueError("expected all 12 validation months")
    diagnostic_file = q2_dir / "episode_diagnostics.parquet"
    if _sha256(diagnostic_file) != q2_manifest["files"]["episode_diagnostics.parquet"]["sha256"]:
        raise ValueError("Q2 episode diagnostics changed")
    episodes = pq.read_table(diagnostic_file,
                             columns=["split", "target_episode_id", "sensor_type"]).to_pandas()
    episodes = episodes.loc[episodes.split == "validation"]
    if episodes.target_episode_id.duplicated().any():
        raise ValueError("duplicate episode in full denominator")
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        positives = db.execute("""SELECT channel_id,prediction_time,sensor_type,
            target_episode_id,label_available_at FROM read_parquet(?,hive_partitioning=false)
            WHERE target=1 ORDER BY channel_id,prediction_time""", [scores]).fetch_df()
    if positives[list(POSITIVE_COLUMNS)].isna().any().any():
        raise ValueError("positive prediction lacks episode attribution")
    if positives.duplicated(["channel_id", "prediction_time"]).any():
        raise ValueError("duplicate channel-hour in saved predictions")
    if not positives.target_episode_id.isin(episodes.target_episode_id).all():
        raise ValueError("saved positive refers to an unassigned episode")
    if not positives.groupby("target_episode_id")[[
        "channel_id", "sensor_type", "label_available_at"
    ]].nunique().eq(1).all().all():
        raise ValueError("episode attribution differs between positive hours")
    if not positives.groupby("channel_id").sensor_type.nunique().eq(1).all():
        raise ValueError("sensor type changes within a channel")
    lead = positives.label_available_at - positives.prediction_time
    if not (lead.gt(timedelta(0)) & lead.le(timedelta(hours=24))).all():
        raise ValueError("positive hour outside prediction horizon")
    span = positives.groupby("target_episode_id").prediction_time.agg(["min", "max"])
    if not (span["max"] - span["min"]).lt(COOLDOWN).all():
        raise ValueError("episode exceeds cooldown; point scheduling is not exact")

    channel_rows = []
    capacity_by_type: Counter[str] = Counter()
    for channel, group in positives.groupby("channel_id", sort=True):
        optimal, greedy = channel_capacity(group.prediction_time.tolist())
        if optimal != greedy:
            raise ValueError("independent scheduling methods disagree")
        kind = str(group.sensor_type.iloc[0])
        capacity_by_type[kind] += optimal
        channel_rows.append({
            "channel_id": channel, "sensor_type": kind,
            "positive_hours": len(group),
            "available_episodes": group.target_episode_id.nunique(),
            "maximum_matches": optimal,
        })
    full_by_type = episodes.groupby("sensor_type").target_episode_id.nunique()
    available_by_type = positives.groupby("sensor_type").target_episode_id.nunique()
    total = episodes.target_episode_id.nunique()
    available = positives.target_episode_id.nunique()
    maximum = sum(capacity_by_type.values())
    needed = total // 2 + 1
    report = {
        "schema_version": "q2-b-independent-oracle-review-v1",
        "source_experiment_manifest_sha256": _sha256(experiment / "manifest.json"),
        "source_q2_manifest_sha256": _sha256(q2_dir / "manifest.json"),
        "full_episodes": total,
        "positive_hours": len(positives),
        "available_episodes": available,
        "admission_only_ceiling": available / total,
        "maximum_with_24h_cooldown": maximum,
        "full_recall_ceiling": maximum / total,
        "minimum_matches_for_recall_above_half": needed,
        "remaining_oracle_slack": maximum - needed,
        "minimum_fraction_of_oracle_capacity": needed / maximum,
        "by_type": [
            {"sensor_type": str(kind), "full_episodes": int(count),
             "available_episodes": int(available_by_type.get(kind, 0)),
             "maximum_matches": int(capacity_by_type.get(str(kind), 0))}
            for kind, count in full_by_type.sort_index().items()
        ],
        "checks": {
            "score_files_hashed": len(scores),
            "every_episode_span_below_cooldown": True,
            "backward_dp_equals_greedy_on_every_channel": True,
        },
        "limitations": [
            "Oracle uses future labels; this is a capacity bound, not model quality.",
            "Unknown-label hours are absent, as in the published B evaluation.",
            "Fixed hourly grid and empty cooldown state at validation start are assumed.",
            "Individual unselected episodes are not necessarily unreachable in every optimal schedule.",
        ],
    }
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if pending.exists():
        raise FileExistsError(pending)
    pending.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(channel_rows), pending / "channel_capacity.parquet",
                   compression="zstd")
    (pending / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    manifest = {"schema_version": report["schema_version"],
                "report_sha256": _sha256(pending / "report.json"),
                "channel_capacity_sha256": _sha256(pending / "channel_capacity.parquet")}
    (pending / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    pending.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "q2-dir", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    result = verify(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in (
        "full_episodes", "available_episodes", "maximum_with_24h_cooldown",
        "full_recall_ceiling")}), flush=True)


if __name__ == "__main__":
    main()
