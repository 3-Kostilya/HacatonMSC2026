"""Retrospective raw evidence for Q2 quality losses, never a new admission rule."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
import time

import duckdb
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_quality_improvement_a import expected_months, quoted, safe_path
from analysis.build_sparse_population_a import write_json
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import read_json, sha256
from stage1.features.hourly import HourlyConfig
from stage1.features.sparse_admission import BLOCKING_QA
from stage1.state_labeling.operational import registered_state_effect
from stage1.state_labeling.rules import KNOWN_SENSOR_TYPES
from stage1.value_quality import assess_value


VERSION = "q2-a-quality-blockers-diagnostic-v1"
TEMPERATURE_RANGE_TEXT = "В норме от +3 до +40"
CANDIDATE_CATEGORIES = frozenset(
    {
        "equivalent_numeric_format_candidate",
        "numeric_norma_companion_candidate",
        "temperature_range_companion_candidate",
    }
)
FLAGS = HourlyConfig().excluded_quality_flags


def classify_group(messages):
    """Candidate compatibility is a review question, not permission to use rows."""
    types = {m["sensor_type"] for m in messages}
    numbers = {m["value_numeric"] for m in messages if m["value_numeric"] is not None}
    texts = {m["value_state"] for m in messages if m["value_state"] is not None}
    alarms = {m["alarm"] for m in messages}
    variants = {(m["value_raw"], m["alarm"]) for m in messages}
    flags = {f for m in messages for f in m["excluded_flags"]}
    qa = {
        assess_value(m["sensor_type"], m["value_raw"], m["value_numeric"]).category
        for m in messages
    }
    effects = {
        registered_state_effect(m["sensor_type"], m["value_state"], m["alarm"]) for m in messages
    }
    contradiction = "fault" in effects and "normal" in effects
    protected = []
    if len(types) != 1 or not types.issubset(KNOWN_SENSOR_TYPES):
        category = "unknown_or_conflicting_type"
    elif contradiction:
        category = "registered_fault_and_normal_same_second"
    elif flags & {"nonfinite_numeric", "invalid_timestamp"}:
        category = "nonfinite_or_invalid_time"
    elif qa & BLOCKING_QA:
        category = "qa_technical_artifact_or_code"
    elif len(numbers) > 1:
        category = "multiple_numeric_values_unordered"
    elif len(variants) <= 1:
        category = "single_variant_visible_in_full_archive"
    elif not texts and len(numbers) == 1 and len(alarms) == 1:
        category = "equivalent_numeric_format_candidate"
    elif len(numbers) == 1 and texts and alarms == {False}:
        sensor_type = next(iter(types))
        number = next(iter(numbers))
        if (
            sensor_type == "Датчик температуры"
            and texts.issubset({"Норма", TEMPERATURE_RANGE_TEXT})
            and TEMPERATURE_RANGE_TEXT in texts
        ):
            category = (
                "temperature_range_companion_candidate"
                if 3 <= number <= 40
                else "numeric_temperature_range_disagreement"
            )
        elif (
            texts == {"Норма"}
            and sensor_type in {"Датчик температуры", "Газовый датчик"}
            and not qa.intersection({"gas_negative_reading", "gas_alarm_level_candidate"})
        ):
            category = "numeric_norma_companion_candidate"
        else:
            category = "numeric_and_text_unresolved"
    elif not numbers and len(texts) == 1 and len(alarms) > 1:
        category = "same_text_alarm_disagreement"
    elif len(texts) > 1:
        category = "multiple_text_states_unordered"
    else:
        category = "other_variant_disagreement"
    if contradiction:
        protected.append("registered_fault_and_normal_same_second")
    protected.extend(sorted(flags & {"nonfinite_numeric", "invalid_timestamp"}))
    protected.extend(sorted(qa & BLOCKING_QA))
    return {
        "pair_category": category,
        "compatibility_candidate_requires_review": category in CANDIDATE_CATEGORIES,
        "registered_state_contradiction": contradiction,
        "multiple_numeric_values": len(numbers) > 1,
        "alarm_difference": len(alarms) > 1,
        "distinct_raw_alarm_variants": len(variants),
        "protected_evidence": sorted(set(protected)),
        "excluded_rows": sum(m["excluded_rows"] for m in messages),
    }


def merged_windows(positive):
    """Union diagnostic windows to avoid repeating overlapping hours in raw scans."""
    times = defaultdict(list)
    for row in positive:
        times[row["channel_id"]].append(row["prediction_time"])
    result = []
    for channel, values in sorted(times.items()):
        merged = []
        for at in sorted(set(values)):
            lower = at - timedelta(hours=24)
            if merged and lower <= merged[-1][1]:
                merged[-1] = merged[-1][0], at
            else:
                merged.append((lower, at))
        result.extend({"channel_id": channel, "lower": low, "upper": high} for low, high in merged)
    return result


def mechanical_hour(row):
    """Keep EVERY other guard, including excluded status and first-history guards."""
    return row["admission_status"] == "eligible" or (
        row["admission_status"] == "unknown"
        and set(row["admission_reasons"]) == {"quality_exclusions_24h"}
        and row["matched_excluded_rows_24h"] > 0
        and row["noncandidate_groups_24h"] == 0
    )


def match_hours(database, positive, lightweight):
    """Count complete groups in (t-24h,t], including previous-month records."""
    database.register("classified", pa.Table.from_pylist(lightweight))
    database.execute(
        "CREATE TEMP TABLE prefix AS SELECT channel_id,timestamp,"
        "CAST(SUM(excluded_rows) OVER w AS BIGINT) AS excluded_count,"
        "CAST(SUM(noncandidate) OVER w AS BIGINT) AS noncandidate_count "
        "FROM classified WINDOW w AS (PARTITION BY channel_id ORDER BY timestamp)"
    )
    database.register("positive", pa.Table.from_pylist(positive))
    sql = (
        "SELECT p.*,CAST(COALESCE(h.excluded_count,0)-COALESCE(l.excluded_count,0) AS BIGINT) "
        "AS matched_excluded_rows_24h,CAST(COALESCE(h.noncandidate_count,0)-COALESCE(l.noncandidate_count,0) "
        "AS BIGINT) AS noncandidate_groups_24h FROM positive p "
        "ASOF LEFT JOIN prefix h ON p.channel_id=h.channel_id AND p.prediction_time>=h.timestamp "
        "ASOF LEFT JOIN prefix l ON p.channel_id=l.channel_id AND "
        "p.prediction_time-INTERVAL '24 hours'>=l.timestamp"
    )
    return database.execute(sql).to_arrow_table().to_pylist()


def audit(*, package, m1_dir, output_dir):
    begun = time.perf_counter()
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or pending.exists():
        raise FileExistsError(output_dir)
    manifest = read_json(package / "manifest.json")
    if [m["month"] for m in manifest["months"]] != expected_months():
        raise ValueError("quality audit requires exactly the accepted 72 train/validation months")
    for name, info in manifest["files"].items():
        if sha256(safe_path(package, name)) != info["sha256"]:
            raise ValueError("Q2 source member differs")
    source_report = read_json(package / "report.json")
    if sha256(m1_dir / "manifest.json") != manifest["source_manifests"]["m1"]:
        raise ValueError("M1 source manifest differs")
    files = []
    for source in source_report["source_m1_files"]:
        file = safe_path(m1_dir, source["file"])
        if (
            any(part in {"year=2021", "year=2026"} for part in file.parts)
            or sha256(file) != source["sha256"]
        ):
            raise ValueError("excluded/test year or mismatched M1 source")
        files.append(str(file))
    if len(files) != 72:
        raise ValueError("M1 audit is not a full 72-month source list")
    positive_path = package / "positive_hour_diagnostics.parquet"
    positive = pq.ParquetFile(positive_path).read().to_pylist()
    if len(positive) != 95217 or Counter(r["split"] for r in positive) != {
        "train": 75629,
        "validation": 19588,
    }:
        raise ValueError("accepted positive-hour population differs")
    if len({(r["channel_id"], r["prediction_time"]) for r in positive}) != len(positive):
        raise ValueError("duplicate positive diagnostic key")
    if any(
        r["target"] != 1 or r["label_status"] != "positive" or r["split_status"] != "assigned"
        for r in positive
    ):
        raise ValueError("unassigned/unknown labels are not positive diagnostic scope")
    windows = merged_windows(positive)
    pending.mkdir(parents=True)
    grouped_counts, row_counts = Counter(), Counter()
    examples = defaultdict(list)
    with duckdb.connect(config={"temp_directory": str(pending / "db-spill")}) as db:
        db.execute("SET memory_limit='3GB'")
        db.execute("SET threads=2")
        db.execute("SET preserve_insertion_order=false")
        db.register("windows", pa.Table.from_pylist(windows))
        db.execute(
            "CREATE TEMP VIEW raw AS SELECT row_id,source,source_row,event_id,channel_id,timestamp,"
            "sensor_type,alarm,value_raw,value_numeric,value_state,quality_flags FROM read_parquet(["
            + ",".join(quoted(f) for f in files)
            + "],hive_partitioning=false) "
            "WHERE year(timestamp)<>2021 AND timestamp<TIMESTAMP '2026-01-01' "
            "AND split_part(replace(source,chr(92),'/'),'/',-1)="
            "'ext-journal-' || CAST(year(timestamp) AS VARCHAR) || '.7z'"
        )
        flag_rows = db.execute(
            "SELECT year(timestamp),sensor_type,flag,COUNT(*) FROM raw,"
            "UNNEST(quality_flags) u(flag) GROUP BY ALL"
        ).fetchall()
        filter_sql = ",".join(quoted(flag) for flag in sorted(FLAGS))
        db.execute(
            "CREATE TEMP TABLE seconds AS SELECT DISTINCT r.channel_id,r.timestamp "
            "FROM raw r SEMI JOIN windows w ON r.channel_id=w.channel_id "
            "AND r.timestamp>w.lower AND r.timestamp<=w.upper "
            f"WHERE list_has_any(COALESCE(r.quality_flags,[]),[{filter_sql}])"
        )
        print("selected complete quality seconds in positive diagnostic history", flush=True)
        flag_columns = ",".join(
            f"BOOL_OR(list_contains(COALESCE(r.quality_flags,[]),{quoted(flag)})) AS flag_{i}"
            for i, flag in enumerate(sorted(FLAGS))
        )
        db.execute(
            "CREATE TEMP TABLE variants AS SELECT r.channel_id,r.timestamp,r.sensor_type,r.alarm,"
            "r.value_raw,r.value_numeric,r.value_state,COUNT(*) AS rows,"
            f"COUNT(*) FILTER(WHERE list_has_any(COALESCE(r.quality_flags,[]),[{filter_sql}])) AS excluded_rows,"
            "arg_min(struct_pack(row_id:=r.row_id,source:=r.source,source_row:=r.source_row,event_id:=r.event_id),"
            "r.row_id,3) AS source_examples," + flag_columns + " FROM raw r SEMI JOIN seconds s "
            "USING(channel_id,timestamp) GROUP BY ALL"
        )
        flag_list = ",".join(
            f"CASE WHEN flag_{i} THEN {quoted(flag)} END" for i, flag in enumerate(sorted(FLAGS))
        )
        db.execute(
            "CREATE TEMP TABLE groups AS SELECT channel_id,timestamp,LIST(struct_pack("
            "sensor_type:=sensor_type,alarm:=alarm,value_raw:=value_raw,value_numeric:=value_numeric,"
            "value_state:=value_state,rows:=rows,excluded_rows:=excluded_rows,source_examples:=source_examples,"
            f"excluded_flags:=list_filter([{flag_list}],x->x IS NOT NULL))) AS messages "
            "FROM variants GROUP BY channel_id,timestamp ORDER BY channel_id,timestamp"
        )
        reader = db.execute("SELECT * FROM groups").to_arrow_reader(batch_size=5000)
        result_schema = pa.schema(
            [
                *reader.schema,
                pa.field("pair_category", pa.string()),
                pa.field("compatibility_candidate_requires_review", pa.bool_()),
                pa.field("registered_state_contradiction", pa.bool_()),
                pa.field("multiple_numeric_values", pa.bool_()),
                pa.field("alarm_difference", pa.bool_()),
                pa.field("distinct_raw_alarm_variants", pa.int64()),
                pa.field("protected_evidence", pa.list_(pa.string())),
                pa.field("excluded_rows", pa.int64()),
                pa.field("sensor_type", pa.string()),
            ]
        )
        group_file = pending / "quality_groups.parquet"
        writer = None
        lightweight = []
        try:
            for batch in reader:
                enriched = []
                for row in batch.to_pylist():
                    result = {**row, **classify_group(row["messages"])}
                    kind = row["messages"][0]["sensor_type"] or "<unknown>"
                    result["sensor_type"] = kind
                    enriched.append(result)
                    lightweight.append(
                        {
                            "channel_id": row["channel_id"],
                            "timestamp": row["timestamp"],
                            "pair_category": result["pair_category"],
                            "excluded_rows": result["excluded_rows"],
                            "noncandidate": int(
                                not result["compatibility_candidate_requires_review"]
                            ),
                        }
                    )
                    key = (row["timestamp"].year, kind, result["pair_category"])
                    grouped_counts[key] += 1
                    row_counts[key] += result["excluded_rows"]
                    if len(examples[key]) < 3:
                        examples[key].append(
                            {
                                "channel_id": row["channel_id"],
                                "timestamp": row["timestamp"].isoformat(),
                                "values": sorted({m["value_raw"] for m in row["messages"]}),
                                "row_ids": [
                                    e["row_id"]
                                    for m in row["messages"]
                                    for e in m["source_examples"]
                                ],
                            }
                        )
                table = pa.Table.from_pylist(enriched, schema=result_schema)
                if writer is None:
                    writer = pq.ParquetWriter(group_file, table.schema, compression="zstd")
                writer.write_table(table.cast(writer.schema))
        finally:
            if writer is not None:
                writer.close()
        if not lightweight:
            raise ValueError("no quality groups found in accepted diagnostic scope")
        hours = match_hours(db, positive, lightweight)
        if any(r["matched_excluded_rows_24h"] != r["excluded_quality_count_24h"] for r in hours):
            raise ValueError("raw exclusion counts differ from causal Q2 snapshot")
        for row in hours:
            row["mechanical_candidate_review_scenario"] = mechanical_hour(row)
        pq.write_table(
            pa.Table.from_pylist(hours),
            pending / "positive_hour_quality.parquet",
            compression="zstd",
        )
        episodes = {}
        for row in hours:
            item = episodes.setdefault(
                (row["split"], row["target_episode_id"]),
                {
                    "split": row["split"],
                    "target_episode_id": row["target_episode_id"],
                    "sensor_type": row["sensor_type"],
                    "channel_id": row["channel_id"],
                    "positive_hours": 0,
                    "current_eligible_hours": 0,
                    "mechanical_candidate_hours": 0,
                    "quality_hours": 0,
                    "only_candidate_quality_hours": 0,
                    "noncandidate_quality_hours": 0,
                    "other_reason_hours": 0,
                },
            )
            item["positive_hours"] += 1
            item["current_eligible_hours"] += int(row["admission_status"] == "eligible")
            item["mechanical_candidate_hours"] += int(row["mechanical_candidate_review_scenario"])
            item["quality_hours"] += int(row["matched_excluded_rows_24h"] > 0)
            item["only_candidate_quality_hours"] += int(
                row["matched_excluded_rows_24h"] > 0 and not row["noncandidate_groups_24h"]
            )
            item["noncandidate_quality_hours"] += int(row["noncandidate_groups_24h"] > 0)
            item["other_reason_hours"] += int(
                bool(set(row["admission_reasons"]) - {"quality_exclusions_24h"})
            )
        episode_rows = list(episodes.values())
        pq.write_table(
            pa.Table.from_pylist(episode_rows),
            pending / "episode_quality.parquet",
            compression="zstd",
        )
    summary = defaultdict(Counter)
    by_type = defaultdict(lambda: defaultdict(Counter))
    for item in episode_rows:
        current = item["current_eligible_hours"] > 0
        metrics = {
            "all_episodes": 1,
            "current_available": int(current),
            "mechanical_candidate_review_available": int(item["mechanical_candidate_hours"] > 0),
            "missed_quality_on_every_hour": int(
                not current and item["quality_hours"] == item["positive_hours"]
            ),
            "missed_noncandidate_quality_on_every_hour": int(
                not current and item["noncandidate_quality_hours"] == item["positive_hours"]
            ),
            "missed_with_candidate_only_quality_hour": int(
                not current and item["only_candidate_quality_hours"] > 0
            ),
        }
        summary[item["split"]].update(metrics)
        by_type[item["split"]][item["sensor_type"]].update(metrics)
    for split in ("train", "validation"):
        if (
            summary[split]["all_episodes"]
            != source_report["label_audit"]["by_split"][split]["all_episodes"]
            or summary[split]["current_available"]
            != source_report["label_audit"]["by_split"][split]["candidate_available_episodes"]
        ):
            raise ValueError("raw audit episode denominator/old availability differs")
    report = {
        "schema_version": VERSION,
        "status": "raw_quality_diagnostics_complete_no_rules_changed",
        "source_q2_manifest_sha256": sha256(package / "manifest.json"),
        "source_m1_manifest_sha256": manifest["source_manifests"]["m1"],
        "source_m1_files": source_report["source_m1_files"],
        "expected_months": expected_months(),
        "positive_hours": len(hours),
        "merged_diagnostic_windows": len(windows),
        "classified_quality_seconds": len(lightweight),
        "raw_count_mismatches": 0,
        "by_split": {k: dict(v) for k, v in summary.items()},
        "by_type": {s: {k: dict(v) for k, v in types.items()} for s, types in by_type.items()},
        "full_source_flag_rows": [
            {"year": y, "sensor_type": k, "flag": f, "rows": n}
            for y, k, f, n in sorted(flag_rows, key=lambda x: (x[0], str(x[1]), x[2]))
        ],
        "quality_pair_groups": [
            {
                "year": y,
                "sensor_type": k,
                "pair_category": c,
                "seconds": n,
                "excluded_rows": row_counts[y, k, c],
                "examples": examples[y, k, c],
            }
            for (y, k, c), n in sorted(grouped_counts.items())
        ],
        "candidate_categories_require_b_review": sorted(CANDIDATE_CATEGORIES),
        "retrospective_diagnostics_not_features_or_training_index": True,
        "mechanical_scenario_not_causal_recalculation": True,
        "admission_rules_changed": False,
        "labels_changed": False,
        "frozen_r6_changed": False,
        "test_events_read": False,
        "physical_failure_claim": False,
        "limitations": [
            "Merged windows and diagnostics use existing future positive labels; never use their keys as a causal index.",
            "All pairing classes are diagnostic; numeric differences can reflect sub-second messages lost to timestamp resolution.",
            "Candidate compatibility is not approved. Removing only the quality reason does not recompute usable history, state or features.",
            "Single visible variants may be flagged by records outside full-archive scope; never clear them automatically.",
        ],
        "code_lf_sha256": frozen_rule_sha256(Path(__file__)),
        "resources": {
            "elapsed_seconds": round(time.perf_counter() - begun, 3),
            "peak_working_set_bytes": getattr(psutil.Process().memory_info(), "peak_wset", 0),
            "duckdb_memory_limit": "3GB",
            "duckdb_threads": 2,
        },
    }
    write_json(pending / "report.json", report)
    write_json(
        pending / "manifest.json",
        {
            "schema_version": VERSION,
            "status": report["status"],
            "source_q2_manifest_sha256": report["source_q2_manifest_sha256"],
            "files": {
                f.name: {"sha256": sha256(f), "bytes": f.stat().st_size}
                for f in (
                    pending / "quality_groups.parquet",
                    pending / "positive_hour_quality.parquet",
                    pending / "episode_quality.parquet",
                    pending / "report.json",
                )
            },
        },
    )
    pending.rename(output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "m1-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = audit(package=args.package, m1_dir=args.m1_dir, output_dir=args.output_dir)
    print(
        {
            k: report[k]
            for k in ("classified_quality_seconds", "positive_hours", "by_split", "resources")
        }
    )


if __name__ == "__main__":
    main()
