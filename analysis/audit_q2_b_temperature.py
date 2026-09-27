"""Independent bounded review of the temperature-channel Q2 bottleneck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from analysis.train_r4_discrete_baselines import read_json, sha256


def run(*, q2_dir: Path, m1_dir: Path, b2_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    q2 = read_json(q2_dir / "report.json")
    manifest = read_json(q2_dir / "manifest.json")
    if (q2["source_manifests"]["m1"] != sha256(m1_dir / "manifest.json")
            or q2["source_manifests"]["m1"] != manifest["source_manifests"]["m1"]):
        raise ValueError("M1/Q2 source differs")
    if sha256(b2_dir / "manifest.json") != (
            "acedc29c7b24a9364c5b197ce8501fa9c98a5ece8e6e53fb9f147e5f5a7a22b8"):
        raise ValueError("accepted B2 episode catalog differs")
    source_files = [m1_dir / item["file"] for item in q2["source_m1_files"]
                    if item["file"].startswith("clean/year=2025/")]
    if len(source_files) != 12:
        raise ValueError("expected twelve 2025 M1 month files")
    episodes = q2_dir / "episode_diagnostics.parquet"
    positive = q2_dir / "positive_hour_diagnostics.parquet"
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        channels = db.execute("""SELECT channel_id,COUNT(*) episodes,
            SUM((candidate_hours>0)::INT) available,
            SUM((candidate_hours=0)::INT) unavailable,
            SUM(positive_hours) positive_hours
            FROM read_parquet(?) WHERE split='validation'
            AND sensor_type='Датчик температуры'
            GROUP BY channel_id ORDER BY episodes DESC""", [str(episodes)]).fetchall()
        direct = db.execute("""WITH per_second AS (
            SELECT timestamp,
                BOOL_OR(value_state='Норма') AS normal,
                BOOL_OR(value_state='Неисправен') AS faulty,
                COUNT(*) AS messages
            FROM read_parquet(?,hive_partitioning=false)
            WHERE channel_id='228571'
              AND split_part(replace(source,chr(92),'/'),'/',-1)
                  ='ext-journal-2025.7z'
            GROUP BY timestamp)
            SELECT COUNT(*) seconds_seen,
                COUNT(*) FILTER(WHERE normal AND faulty) simultaneous_seconds,
                SUM(messages) FILTER(WHERE normal AND faulty) simultaneous_messages,
                MIN(timestamp) FILTER(WHERE normal AND faulty) first_conflict,
                MAX(timestamp) FILTER(WHERE normal AND faulty) last_conflict
            FROM per_second""", [[str(path) for path in source_files]]).fetchone()
        blocker = db.execute("""SELECT COUNT(*) positive_hours,
            COUNT(*) FILTER(WHERE admission_status='eligible') eligible_hours,
            COUNT(*) FILTER(WHERE list_contains(admission_reasons,
                'quality_exclusions_24h')) quality_blocked_hours
            FROM read_parquet(?) WHERE split='validation' AND channel_id='228571'""",
                             [str(positive)]).fetchone()
        episode_rows = db.execute("""SELECT b.start_at,b.end_at,b.onset_status,
            b.end_status,b.fault_message_count,b.uncertain_intervening_state
            FROM read_parquet(?) b JOIN read_parquet(?) q
            ON b.episode_id=q.target_episode_id
            WHERE q.split='validation' AND q.channel_id='228571'
            ORDER BY b.start_at""", [str(b2_dir / "registered_state_episodes.parquet"),
                                      str(episodes)]).fetchall()
    if (not channels or channels[0][0] != "228571"
            or channels[0][3] != 406 or sum(row[1] for row in channels) != 408
            or direct[1] <= 0 or len(episode_rows) != 406):
        raise ValueError("temperature episode concentration or direct conflict differs")
    gaps = [(after[0] - before[0]).total_seconds() / 3600
            for before, after in zip(episode_rows, episode_rows[1:], strict=False)]
    durations = [(end - start).total_seconds() / 3600
                 for start, end, *_ in episode_rows if end is not None]
    report = {
        "schema_version": "q2-b-temperature-trace-audit-v1",
        "q2_manifest_sha256": sha256(q2_dir / "manifest.json"),
        "m1_manifest_sha256": sha256(m1_dir / "manifest.json"),
        "b2_manifest_sha256": sha256(b2_dir / "manifest.json"),
        "validation_temperature_channels": [
            {"channel_id": c, "episodes": n, "available": a,
             "unavailable": u, "positive_hours": h} for c, n, a, u, h in channels],
        "channel_228571_2025": {
            "seconds_seen": direct[0], "simultaneous_normal_fault_seconds": direct[1],
            "messages_at_simultaneous_seconds": direct[2],
            "first_simultaneous_second": direct[3].isoformat(),
            "last_simultaneous_second": direct[4].isoformat(),
            "assigned_positive_hours": blocker[0],
            "eligible_positive_hours": blocker[1],
            "quality_blocked_positive_hours": blocker[2],
        },
        "channel_228571_episodes": {
            "count": len(episode_rows),
            "onset_statuses": sorted({row[2] for row in episode_rows}),
            "end_statuses": sorted({row[3] for row in episode_rows}),
            "ended_count": len(durations),
            "one_fault_message_count": sum(row[4] == 1 for row in episode_rows),
            "uncertain_intervening_count": sum(row[5] for row in episode_rows),
            "start_gap_under_1h": sum(gap < 1 for gap in gaps),
            "start_gap_under_24h": sum(gap < 24 for gap in gaps),
            "median_start_gap_hours": sorted(gaps)[len(gaps) // 2],
            "median_ended_duration_hours": (
                sorted(durations)[len(durations) // 2] if durations else None),
        },
        "decision": "Keep simultaneous normal/fault as ambiguous; do not reorder or merge episodes from row order.",
        "test_data_read": False,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q2-dir", required=True, type=Path)
    parser.add_argument("--m1-dir", required=True, type=Path)
    parser.add_argument("--b2-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    result = run(**vars(parser.parse_args()))
    print(json.dumps(result["channel_228571_2025"], default=str), flush=True)


if __name__ == "__main__":
    main()
