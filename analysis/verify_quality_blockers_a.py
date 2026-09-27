"""Independent range-join counts, source traces and transfer proof for Q2 diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
import zipfile

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.audit_quality_blockers_a import classify_group, mechanical_hour
from analysis.build_quality_improvement_a import quoted, safe_path
from analysis.build_sparse_population_a import write_json
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import read_json, sha256
from stage1.features.hourly import HourlyConfig


def verify(*, package, q2_package, m1_dir, output, handoff_zip=None):
    begun = time.perf_counter()
    if output.exists() or (handoff_zip is not None and handoff_zip.exists()):
        raise FileExistsError("verification/transfer output already exists")
    source_sha = sha256(q2_package / "manifest.json")
    manifest_sha = sha256(package / "manifest.json")
    manifest = read_json(package / "manifest.json")
    report = read_json(package / "report.json")
    if manifest["source_q2_manifest_sha256"] != source_sha:
        raise ValueError("diagnostic Q2 provenance differs")
    if sha256(m1_dir / "manifest.json") != report["source_m1_manifest_sha256"]:
        raise ValueError("M1 manifest differs")
    for name, info in manifest["files"].items():
        path = safe_path(package, name)
        if sha256(path) != info["sha256"] or path.stat().st_size != info["bytes"]:
            raise ValueError("diagnostic file checksum differs")
    groups_path = package / "quality_groups.parquet"
    hours_path = package / "positive_hour_quality.parquet"
    episodes_path = package / "episode_quality.parquet"
    original_path = q2_package / "positive_hour_diagnostics.parquet"
    samples = []
    seen_strata = Counter()
    groups = pq.ParquetFile(groups_path).read().to_pylist()
    for group in groups:
        recomputed = classify_group(group["messages"])
        if any(group[key] != value for key, value in recomputed.items()):
            raise ValueError("stored classification differs from complete messages")
        stratum = (group["sensor_type"], group["pair_category"])
        if seen_strata[stratum] < 2:
            samples.append({"channel_id": group["channel_id"], "timestamp": group["timestamp"]})
            seen_strata[stratum] += 1
    hours = pq.ParquetFile(hours_path).read().to_pylist()
    if any(row["mechanical_candidate_review_scenario"] != mechanical_hour(row) for row in hours):
        raise ValueError("mechanical diagnostic scenario differs")
    episode_rows = pq.ParquetFile(episodes_path).read().to_pylist()
    episodes = {(r["split"], r["target_episode_id"]): r for r in episode_rows}
    if len(episodes) != len(episode_rows):
        raise ValueError("duplicate episode summary key")
    files = []
    for entry in report["source_m1_files"]:
        path = safe_path(m1_dir, entry["file"])
        if (
            any(part in {"year=2021", "year=2026"} for part in path.parts)
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError("excluded/test or changed source file")
        files.append(path)
    if len(files) != 72:
        raise ValueError("incomplete raw source scope")
    flag_list = ",".join(quoted(f) for f in sorted(HourlyConfig().excluded_quality_flags))
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        for name, path in (
            ("h", hours_path),
            ("g", groups_path),
            ("o", original_path),
            ("e", episodes_path),
        ):
            db.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet({quoted(str(path))},hive_partitioning=false)"
            )
        fields = ",".join(
            '"' + name.replace('"', '""') + '"'
            for name in pq.ParquetFile(original_path).schema_arrow.names
        )
        for left, right in (("h", "o"), ("o", "h")):
            if db.execute(
                f"SELECT COUNT(*) FROM (SELECT {fields} FROM {left} EXCEPT ALL SELECT {fields} FROM {right})"
            ).fetchone()[0]:
                raise ValueError("source keys, labels, reasons or snapshot fields changed")
        db.execute(
            "CREATE TEMP TABLE direct AS SELECT h.channel_id,h.prediction_time,"
            "CAST(COALESCE(SUM(g.excluded_rows),0) AS BIGINT) AS excluded_rows,"
            "CAST(COALESCE(SUM(CAST(NOT g.compatibility_candidate_requires_review AS BIGINT)),0) AS BIGINT) AS noncandidate "
            "FROM h LEFT JOIN g ON h.channel_id=g.channel_id "
            "AND g.timestamp>h.prediction_time-INTERVAL '24 hours' AND g.timestamp<=h.prediction_time GROUP BY ALL"
        )
        if db.execute(
            "SELECT COUNT(*) FROM h JOIN direct d USING(channel_id,prediction_time) "
            "WHERE h.matched_excluded_rows_24h<>d.excluded_rows OR h.noncandidate_groups_24h<>d.noncandidate"
        ).fetchone()[0]:
            raise ValueError("independent direct-window join differs from prefix counts")
        db.execute(
            "CREATE TEMP TABLE hour_categories AS SELECT DISTINCT h.split,h.target_episode_id,"
            "h.channel_id,h.prediction_time,h.sensor_type,g.pair_category FROM h JOIN g "
            "ON h.channel_id=g.channel_id AND g.timestamp>h.prediction_time-INTERVAL '24 hours' "
            "AND g.timestamp<=h.prediction_time"
        )
        category_hours = db.execute(
            "SELECT split,target_episode_id,pair_category,COUNT(*) "
            "FROM hour_categories GROUP BY ALL"
        ).fetchall()
        temperature_channels = (
            db.execute(
                "SELECT e.split,e.channel_id,COUNT(*) AS episodes,"
                "COUNT(*) FILTER(WHERE current_eligible_hours>0) AS current_available,"
                "COUNT(*) FILTER(WHERE mechanical_candidate_hours>0) AS mechanical_review_available,"
                "MIN(positive_hours) AS min_positive_hours,MAX(positive_hours) AS max_positive_hours,"
                "MIN(times.first_hour) AS first_hour,MAX(times.last_hour) AS last_hour "
                "FROM e JOIN (SELECT split,target_episode_id,MIN(prediction_time) AS first_hour,"
                "MAX(prediction_time) AS last_hour FROM h GROUP BY ALL) times USING(split,target_episode_id) "
                "WHERE e.sensor_type='Датчик температуры' GROUP BY e.split,e.channel_id "
                "ORDER BY e.split,episodes DESC,e.channel_id"
            )
            .to_arrow_table()
            .to_pylist()
        )
        category_summary = defaultdict(Counter)
        for split, episode_id, category, count in category_hours:
            episode = episodes[split, episode_id]
            if episode["current_eligible_hours"]:
                continue
            category_summary[split, episode["sensor_type"], category].update(
                {
                    "missed_episodes_any_positive_hour": 1,
                    "missed_episodes_every_positive_hour": int(count == episode["positive_hours"]),
                }
            )
        recomputed_episodes = db.execute(
            "SELECT split,target_episode_id,ANY_VALUE(sensor_type),ANY_VALUE(channel_id),"
            "COUNT(*),COUNT(*) FILTER(WHERE admission_status='eligible'),"
            "COUNT(*) FILTER(WHERE mechanical_candidate_review_scenario),"
            "COUNT(*) FILTER(WHERE matched_excluded_rows_24h>0),"
            "COUNT(*) FILTER(WHERE matched_excluded_rows_24h>0 AND noncandidate_groups_24h=0),"
            "COUNT(*) FILTER(WHERE noncandidate_groups_24h>0),"
            "COUNT(*) FILTER(WHERE len(list_filter(admission_reasons,x->x<>'quality_exclusions_24h'))>0) "
            "FROM h GROUP BY split,target_episode_id"
        ).fetchall()
        metrics = (
            "sensor_type",
            "channel_id",
            "positive_hours",
            "current_eligible_hours",
            "mechanical_candidate_hours",
            "quality_hours",
            "only_candidate_quality_hours",
            "noncandidate_quality_hours",
            "other_reason_hours",
        )
        for row in recomputed_episodes:
            if tuple(episodes[row[0], row[1]][key] for key in metrics) != row[2:]:
                raise ValueError("independent SQL episode aggregation differs")
        db.register("samples", pa.Table.from_pylist(samples))
        db.execute(
            "CREATE TEMP TABLE traces AS SELECT r.channel_id,r.timestamp,r.sensor_type,r.alarm,"
            "r.value_raw,r.value_numeric,r.value_state,COUNT(*) AS rows,"
            "LIST(struct_pack(row_id:=r.row_id,source:=r.source,source_row:=r.source_row,event_id:=r.event_id)) AS source_records,"
            "list_distinct(flatten(LIST(COALESCE(r.quality_flags,[])))) AS flags,"
            f"COUNT(*) FILTER(WHERE list_has_any(COALESCE(r.quality_flags,[]),[{flag_list}])) AS excluded_rows "
            "FROM read_parquet(["
            + ",".join(quoted(str(f)) for f in files)
            + "],hive_partitioning=false) r "
            "SEMI JOIN samples s USING(channel_id,timestamp) "
            "WHERE split_part(replace(r.source,chr(92),'/'),'/',-1)="
            "'ext-journal-'||CAST(year(r.timestamp) AS VARCHAR)||'.7z' GROUP BY ALL"
        )
        trace_rows = db.execute("SELECT * FROM traces").to_arrow_table().to_pylist()
        trace_signature = Counter(
            tuple(
                row[k]
                for k in (
                    "channel_id",
                    "timestamp",
                    *metrics[:1],
                    "alarm",
                    "value_raw",
                    "value_numeric",
                    "value_state",
                    "rows",
                    "excluded_rows",
                )
            )
            for row in trace_rows
        )
        selected_keys = {(row["channel_id"], row["timestamp"]) for row in samples}
        stored_signature = Counter(
            (
                group["channel_id"],
                group["timestamp"],
                message["sensor_type"],
                message["alarm"],
                message["value_raw"],
                message["value_numeric"],
                message["value_state"],
                message["rows"],
                message["excluded_rows"],
            )
            for group in groups
            if (group["channel_id"], group["timestamp"]) in selected_keys
            for message in group["messages"]
        )
        if trace_signature != stored_signature:
            raise ValueError("raw M1 complete-group trace differs")
        source_fields = ("row_id", "source", "source_row", "event_id")
        raw_variants = {
            tuple(
                row[key]
                for key in (
                    "channel_id",
                    "timestamp",
                    "sensor_type",
                    "alarm",
                    "value_raw",
                    "value_numeric",
                    "value_state",
                )
            ): row
            for row in trace_rows
        }
        for group in groups:
            if (group["channel_id"], group["timestamp"]) not in selected_keys:
                continue
            for message in group["messages"]:
                key = (
                    group["channel_id"],
                    group["timestamp"],
                    *(
                        message[field]
                        for field in (
                            "sensor_type",
                            "alarm",
                            "value_raw",
                            "value_numeric",
                            "value_state",
                        )
                    ),
                )
                raw = raw_variants[key]
                records = {
                    tuple(record[field] for field in source_fields)
                    for record in raw["source_records"]
                }
                if not all(
                    tuple(example[field] for field in source_fields) in records
                    for example in message["source_examples"]
                ):
                    raise ValueError("stored raw provenance example differs")
                if (
                    set(message["excluded_flags"])
                    != set(raw["flags"]) & HourlyConfig().excluded_quality_flags
                ):
                    raise ValueError("sample raw quality flags differ")
    if (
        sha256(q2_package / "manifest.json") != source_sha
        or sha256(package / "manifest.json") != manifest_sha
    ):
        raise ValueError("source/diagnostic manifest changed during verification")
    result = {
        "schema_version": "q2-a-quality-blockers-verification-v1",
        "status": "verified",
        "source_q2_manifest_sha256": source_sha,
        "diagnostic_manifest_sha256": manifest_sha,
        "original_hour_fields_unchanged": True,
        "diagnostic_hours": len(hours),
        "episode_summaries_checked": len(episodes),
        "classified_groups_checked": len(groups),
        "direct_window_count_mismatches": 0,
        "raw_trace_groups": len(samples),
        "raw_trace_variants": len(trace_rows),
        "raw_trace_mismatches": 0,
        "raw_trace_examples_and_flags_checked": True,
        "source_m1_files_checked": len(files),
        "test_events_read": False,
        "code_lf_sha256": frozen_rule_sha256(Path(__file__)),
        "missed_episode_quality_categories": [
            {"split": split, "sensor_type": kind, "pair_category": category, **dict(counts)}
            for (split, kind, category), counts in sorted(category_summary.items())
        ],
        "category_counts_overlap": True,
        "temperature_channel_episodes": [
            {
                key: value.isoformat() if hasattr(value, "isoformat") else value
                for key, value in row.items()
            }
            for row in temperature_channels
        ],
        "limitations": [
            "Positive-hour diagnostics use future labels and are not a causal admission index.",
            "No compatibility rule is accepted and no coverage gain/model metric is claimed.",
        ],
        "elapsed_seconds": round(time.perf_counter() - begun, 3),
    }
    write_json(output, result)
    transfer = None
    if handoff_zip is not None:
        members = {f"{package.name}/{name}": safe_path(package, name) for name in manifest["files"]}
        members[f"{package.name}/manifest.json"] = package / "manifest.json"
        members[f"{package.name}/verification.json"] = output
        with zipfile.ZipFile(
            handoff_zip, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name, path in members.items():
                archive.write(path, name)
        with zipfile.ZipFile(handoff_zip) as archive:
            if archive.testzip() is not None or set(archive.namelist()) != set(members):
                raise ValueError("transfer archive CRC/membership differs")
            import hashlib

            for name, path in members.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != sha256(path):
                    raise ValueError("transfer archive member differs")
        transfer = {
            "path": str(handoff_zip),
            "sha256": sha256(handoff_zip),
            "bytes": handoff_zip.stat().st_size,
            "members": len(members),
        }
    return {
        "verification": {
            k: v for k, v in result.items() if k != "missed_episode_quality_categories"
        },
        "transfer": transfer,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "q2-package", "m1-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--handoff-zip", type=Path)
    args = parser.parse_args()
    print(
        verify(
            package=args.package,
            q2_package=args.q2_package,
            m1_dir=args.m1_dir,
            output=args.output,
            handoff_zip=args.handoff_zip,
        )
    )


if __name__ == "__main__":
    main()
