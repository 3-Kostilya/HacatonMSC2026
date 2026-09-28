"""Audit all coverage losses and materialize label-free delta features for B.

Three PREDECLARED research ablations, not a threshold/model search or a
production change. Existing Q2 features are immutable and referenced, not
copied. Labels are read only after ALL causal month payloads are published.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
import shutil
import time

import duckdb
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_quality_improvement_a import QA_NAMES, counts_sql, create_qa_prefixes
from analysis.build_quality_improvement_a import expected_months, quoted, safe_path
from analysis.build_sparse_population_a import write_json
from analysis.coverage_reentry_a import EVIDENCE_FIELDS, PAST_FIELDS, POLICIES, VERSION
from analysis.coverage_reentry_a import ResearchCoverageStream, policy_sql
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import read_json, sha256


Q2_PIN = "c9775a94bfcff09d1c641b010b93d62e9927209017ccbc7346749f78525de619"
CODE = ("analysis/coverage_reentry_a.py", "analysis/build_coverage_reentry_a.py")


def quality_prefix(db, files):
    """Only bad rows; complete sources. ASOF never consumes future flags."""
    db.execute(
        "CREATE TEMP VIEW scope AS SELECT *,CASE WHEN year(timestamp)<2021 THEN 0 ELSE 1 END "
        "AS archive_segment FROM read_parquet(["
        + ",".join(quoted(f) for f in files)
        + "],hive_partitioning=false) WHERE "
        "timestamp>=TIMESTAMP '2019-01-01' AND timestamp<TIMESTAMP '2026-01-01' "
        "AND year(timestamp)<>2021 AND split_part(replace(source,chr(92),'/'),'/',-1)="
        "'ext-journal-'||CAST(year(timestamp) AS VARCHAR)||'.7z'",
    )
    db.execute(
        "CREATE TEMP TABLE raw_quality AS SELECT channel_id,archive_segment,timestamp,"
        "COUNT(*) AS n,BOOL_OR(list_contains(quality_flags,'channel_time_conflict') AND NOT "
        "list_has_any(quality_flags,['nonfinite_numeric','invalid_timestamp'])) AS conflict,"
        "BOOL_OR(list_has_any(quality_flags,['nonfinite_numeric','invalid_timestamp'])) AS hard "
        "FROM scope WHERE list_has_any(COALESCE(quality_flags,[]),"
        "['channel_time_conflict','nonfinite_numeric','invalid_timestamp']) GROUP BY ALL"
    )
    db.execute(
        "CREATE TEMP TABLE raw_quality_prefix AS SELECT channel_id,archive_segment,timestamp,"
        "CAST(SUM(n) OVER w AS BIGINT) AS count,"
        "MAX(CASE WHEN conflict THEN timestamp END) OVER w AS last_conflict_at,"
        "MAX(CASE WHEN hard THEN timestamp END) OVER w AS last_hard_quality_at "
        "FROM raw_quality WINDOW w AS (PARTITION BY channel_id,archive_segment ORDER BY timestamp)"
    )
    return db.execute("SELECT COALESCE(SUM(n),0) FROM raw_quality").fetchone()[0]


def evidence_sql():
    return (
        "SELECT a.*,COALESCE(hi.count,0)-COALESCE(lo.count,0) AS quality_rows_24h,"
        "hi.last_conflict_at,hi.last_hard_quality_at FROM admission a "
        "ASOF LEFT JOIN raw_quality_prefix hi ON a.channel_id=hi.channel_id AND "
        "a.archive_segment=hi.archive_segment AND a.prediction_time>=hi.timestamp "
        "ASOF LEFT JOIN raw_quality_prefix lo ON a.channel_id=lo.channel_id AND "
        "a.archive_segment=lo.archive_segment AND a.prediction_time-INTERVAL '24 hours'>=lo.timestamp"
    )


def verify_replay(db, changes, files):
    """Hash-selected changed channels; no outcomes determine trace selection."""
    if not changes:
        return {"snapshots": 0, "status": "no_changed_keys_to_replay"}
    channels = sorted({r["channel_id"] for r in changes})
    db.register("replay_channels", pa.table({"channel_id": channels}))
    end = max(r["prediction_time"] for r in changes)
    reader = db.execute(
        "SELECT r.row_id,r.channel_id,r.timestamp,r.alarm,r.value_numeric,r.value_state,"
        "r.sensor_type,r.object_id,r.join_status,r.quality_flags,r.source "
        "FROM read_parquet(?,hive_partitioning=false) r SEMI JOIN replay_channels USING(channel_id) "
        "WHERE r.timestamp<=? ORDER BY timestamp,channel_id,row_id",
        [files, end],
    ).to_arrow_reader(batch_size=25000)
    rows = (r for b in reader for r in b.to_pylist())
    groups = (list(g) for _, g in groupby(rows, lambda r: (r["timestamp"], r["channel_id"])))
    group = next(groups, None)
    stream = ResearchCoverageStream()
    checked, raw_rows, traces = 0, 0, []
    for at, keys in groupby(
        sorted(changes, key=lambda r: r["prediction_time"]), lambda r: r["prediction_time"]
    ):
        expected = list(keys)
        while group is not None and group[0]["timestamp"] <= at:
            stream.observe_records(group)
            raw_rows += len(group)
            group = next(groups, None)
        actual = {
            r["channel_id"]: r for r in stream.evaluate(at, [r["channel_id"] for r in expected])
        }
        for row in expected:
            got = actual[row["channel_id"]]
            for field in (
                "admission_status",
                "admission_reasons",
                "last_explicit_normal_at",
                "first_usable_at",
                "second_usable_at",
                *EVIDENCE_FIELDS,
            ):
                if got[field] != row[field]:
                    raise ValueError(
                        f"independent raw replay mismatch: {field}, {row['channel_id']}, {at}"
                    )
            for policy in POLICIES:
                if got[policy]["research_status"] != row[policy + "_status"] or (
                    got[policy]["research_reasons"] != row[policy + "_reasons"]
                ):
                    raise ValueError("independent stream differs from SQL policy")
            checked += 1
            traces.append(
                {
                    name: row[name].isoformat() if hasattr(row[name], "isoformat") else row[name]
                    for name in (
                        "channel_id",
                        "prediction_time",
                        "admission_reasons",
                        "last_explicit_normal_at",
                        "last_conflict_at",
                        "last_hard_quality_at",
                        "first_usable_at",
                        "second_usable_at",
                        *[p + "_status" for p in POLICIES],
                    )
                }
            )
    return {"snapshots": checked, "raw_rows": raw_rows, "mismatches": 0, "examples": traces}


def summarize_episodes(rows):
    """Explicit all-episode denominator; overlapping reasons not added as totals."""
    episodes = {}
    for row in rows:
        key = row["split"], row["target_episode_id"]
        item = episodes.setdefault(
            key,
            {
                "split": key[0],
                "target_episode_id": key[1],
                "channel_id": row["channel_id"],
                "sensor_type": row["sensor_type"],
                "positive_hours": 0,
                "base_eligible_hours": 0,
                **{p + "_eligible_hours": 0 for p in POLICIES},
                "reason_sets": set(),
                "per_hour_remaining": [],
            },
        )
        if (item["channel_id"], item["sensor_type"]) != (row["channel_id"], row["sensor_type"]):
            raise ValueError("episode identity/type differs")
        item["positive_hours"] += 1
        item["base_eligible_hours"] += row["admission_status"] == "eligible"
        for policy in POLICIES:
            item[policy + "_eligible_hours"] += row[policy + "_status"] == "eligible"
        item["reason_sets"].add(tuple(row["admission_reasons"]))
        item["per_hour_remaining"].append(set(row["combined_reasons"]))
    totals, by_type = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter))
    result = []
    for item in episodes.values():
        available = item["base_eligible_hours"] > 0
        if available:
            category = "already_available"
        elif item["cold_start_eligible_hours"] and item["after_normal_eligible_hours"]:
            category = "reachable_by_either_single_ablation"
        elif item["cold_start_eligible_hours"]:
            category = "cold_start_only"
        elif item["after_normal_eligible_hours"]:
            category = "later_normal_only"
        elif item["combined_eligible_hours"]:
            category = "needs_both_ablations"
        else:
            category = "still_protected"
        item["coverage_category"] = category
        item["reason_sets"] = [list(r) for r in sorted(item["reason_sets"])]
        remaining = item.pop("per_hour_remaining")
        item["remaining_reasons_any_hour"] = sorted(set.union(*remaining))
        item["remaining_reasons_every_hour"] = sorted(set.intersection(*remaining))
        counts = {
            "all_episodes": 1,
            "base_available": int(available),
            category: 1,
            **{p + "_available": int(item[p + "_eligible_hours"] > 0) for p in POLICIES},
        }
        totals[item["split"]].update(counts)
        by_type[item["split"]][item["sensor_type"]].update(counts)
        result.append(item)
    return (
        result,
        {s: dict(c) for s, c in totals.items()},
        {s: {k: dict(c) for k, c in types.items()} for s, types in by_type.items()},
    )


def build(*, q2_dir, m1_dir, a3_dir, b3_dir, corrections_dir, output_dir, resume_from=None):
    begun = time.perf_counter()
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or pending.exists():
        raise FileExistsError(output_dir)
    if sha256(q2_dir / "manifest.json") != Q2_PIN:
        raise ValueError("requires the accepted immutable Q2 package")
    manifest, report = [read_json(q2_dir / n) for n in ("manifest.json", "report.json")]
    for name, info in manifest["files"].items():
        if sha256(safe_path(q2_dir, name)) != info["sha256"]:
            raise ValueError("Q2 diagnostic/source file differs")
    if [m["month"] for m in manifest["months"]] != expected_months():
        raise ValueError("Q2 months differ")
    pins = manifest["source_manifests"]
    for root, pin in ((m1_dir, pins["m1"]), (a3_dir, pins["a3"]), (b3_dir, pins["b3"])):
        if sha256(root / "manifest.json") != pin:
            raise ValueError("accepted source manifest differs")
    correction = corrections_dir / "feature_corrections.parquet"
    if sha256(correction) != pins["corrections"]:
        raise ValueError("QA corrections differ")
    files = []
    for item in report["source_m1_files"]:
        path = safe_path(m1_dir, item["file"])
        if (
            any(p in {"year=2021", "year=2026"} for p in path.parts)
            or sha256(path) != item["sha256"]
        ):
            raise ValueError("test/excluded year or M1 hash differs")
        files.append(str(path))
    if len(files) != 72:
        raise ValueError("not all 72 M1 months")
    a_chunks = {c["month"]: c for c in read_json(a3_dir / "manifest.json")["chunks"]}
    b_chunks = {c["month"]: c for c in read_json(b3_dir / "manifest.json")["chunks"]}
    allowlist = read_json(q2_dir / "model_feature_allowlist.json")
    base_names, feature_names = allowlist["base_feature_names"], allowlist["feature_names"]
    pending.mkdir(parents=True)
    months, samples, label_proofs = [], [], []
    with duckdb.connect(config={"temp_directory": str(pending / "db-spill")}) as db:
        db.execute("SET threads=1")
        db.execute("SET memory_limit='1500MB'")
        # Arrow registrations can be estimated as one row and select quadratic
        # ASOF nested loops against the full history. Force the equivalent merge.
        db.execute("SET asof_loop_join_threshold=0")
        db.execute("SET preserve_insertion_order=false")
        raw_bad = quality_prefix(db, files)
        print({"raw_quality_rows": raw_bad, "all_72_M1_sources_verified": True}, flush=True)
        if resume_from is None:
            create_qa_prefixes(db, [Path(p) for p in files])
            db.execute("DROP TABLE qa_events")
        db.execute(
            "CREATE TEMP VIEW corrections AS SELECT * FROM read_parquet("
            + quoted(str(correction))
            + ")"
        )
        for month in manifest["months"]:
            label = month["month"]
            folder = pending / f"year={label[:4]}" / f"month={label[5:]}"
            folder.mkdir(parents=True)
            source_folder = q2_dir / Path(month["manifest_file"]).parent
            if sha256(source_folder / "manifest.json") != month["manifest_sha256"]:
                raise ValueError("Q2 monthly manifest differs")
            admission = source_folder / "admission.parquet"
            if sha256(admission) != month["files"][admission.name]["sha256"]:
                raise ValueError("Q2 admission differs")
            if resume_from is not None:
                old_folder = resume_from / Path(month["manifest_file"]).parent
                old_meta = read_json(old_folder / "manifest.json")
                if old_meta["month"] != label or old_meta["source_admission_sha256"] != sha256(
                    admission
                ):
                    raise ValueError("cached causal month source differs")
                a = a_chunks[label]
                feature_source = safe_path(a3_dir, a["features_file"], month=label)
                if sha256(feature_source) != old_meta["source_a3_features_sha256"]:
                    raise ValueError("cached A3 source differs")
                for name, info in old_meta["files"].items():
                    if sha256(old_folder / name) != info["sha256"]:
                        raise ValueError("cached causal month payload differs")
                old_delta = old_folder / "new_admission.parquet"
                table = pq.ParquetFile(old_delta).read()
                incidental = [n for n in ("year", "month") if n in table.column_names]
                if incidental:
                    table = table.drop(incidental)
                delta = folder / old_delta.name
                pq.write_table(table, delta, compression="zstd")
                feature_path = folder / "new_model_features.parquet"
                shutil.copy2(old_folder / feature_path.name, feature_path)
                metadata = {
                    **old_meta,
                    "reused_causal_payload": True,
                    "incidental_hive_columns_removed": incidental,
                    "source_cached_month_manifest_sha256": sha256(old_folder / "manifest.json"),
                    "files": {
                        p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                        for p in (delta, feature_path)
                    },
                }
                write_json(folder / "manifest.json", metadata)
                months.append(
                    {
                        **metadata,
                        "manifest_file": (folder / "manifest.json").relative_to(pending).as_posix(),
                        "manifest_sha256": sha256(folder / "manifest.json"),
                    }
                )
                samples.extend(
                    db.execute(
                        "SELECT * FROM read_parquet(?,hive_partitioning=false) "
                        "QUALIFY ROW_NUMBER() OVER(PARTITION BY sensor_type,can_cold_start,can_reenter "
                        "ORDER BY md5(channel_id),prediction_time)=1",
                        [str(delta)],
                    )
                    .to_arrow_table()
                    .to_pylist()
                )
                print(
                    {
                        "month": label,
                        "reused_and_hash_checked": True,
                        "seconds": round(time.perf_counter() - begun, 1),
                    },
                    flush=True,
                )
                continue
            db.execute(
                "CREATE OR REPLACE TEMP VIEW admission AS SELECT *,"
                "CASE WHEN year(prediction_time)<2021 THEN 0 ELSE 1 END AS archive_segment "
                "FROM read_parquet(" + quoted(str(admission)) + ",hive_partitioning=false)"
            )
            db.execute("CREATE OR REPLACE TEMP TABLE evidence AS " + evidence_sql())
            count, bad = db.execute(
                "SELECT COUNT(*),COUNT(*) FILTER(WHERE quality_rows_24h<>excluded_quality_count_24h "
                "OR first_usable_at>prediction_time OR second_usable_at>prediction_time "
                "OR last_explicit_normal_at>prediction_time OR admission_evidence_through>prediction_time "
                "OR last_conflict_at>prediction_time OR last_hard_quality_at>prediction_time "
                "OR availability_status<>'unknown') FROM evidence"
            ).fetchone()
            if count != month["decision_rows"] or bad:
                raise ValueError("full raw QA reconciliation or causal evidence differs")
            db.execute("CREATE OR REPLACE TEMP TABLE candidates AS " + policy_sql())
            groups = (
                db.execute(
                    "SELECT sensor_type,admission_status,cold_start_status,after_normal_status,"
                    "combined_status,COUNT(*) AS hours FROM candidates GROUP BY ALL"
                )
                .to_arrow_table()
                .to_pylist()
            )
            if any(
                g["admission_status"] == "eligible" and g["combined_status"] != "eligible"
                for g in groups
            ):
                raise ValueError("research extension lost an existing eligible hour")
            changed = folder / "new_admission.parquet"
            db.execute(
                "COPY (SELECT * FROM candidates WHERE admission_status<>'eligible' "
                "AND combined_status='eligible' ORDER BY channel_id,prediction_time) TO "
                + quoted(str(changed))
                + " (FORMAT PARQUET,COMPRESSION ZSTD)"
            )
            # Fixed hash-ranked samples before reading any label or model score.
            samples.extend(
                db.execute(
                    "SELECT * FROM candidates WHERE admission_status<>'eligible' AND combined_status='eligible' "
                    "QUALIFY ROW_NUMBER() OVER(PARTITION BY sensor_type,can_cold_start,can_reenter "
                    "ORDER BY md5(channel_id),prediction_time)=1"
                )
                .to_arrow_table()
                .to_pylist()
            )
            db.execute(
                "CREATE OR REPLACE TEMP TABLE keys AS SELECT channel_id,prediction_time,"
                "archive_segment FROM candidates WHERE admission_status<>'eligible' AND combined_status='eligible'"
            )
            db.execute("CREATE OR REPLACE TEMP TABLE qa_delta AS " + counts_sql())
            a = a_chunks[label]
            feature_source = safe_path(a3_dir, a["features_file"], month=label)
            if sha256(feature_source) != a["features_sha256"]:
                raise ValueError("A3 feature source differs")
            projection = ",".join(
                f'CASE WHEN c.channel_id IS NOT NULL THEN c."{n}" ELSE f."{n}" END AS "{n}"'
                for n in base_names
            )
            db.execute(
                "CREATE OR REPLACE TEMP TABLE feature_delta AS SELECT k.channel_id,k.prediction_time,"
                + projection
                + ","
                + ",".join(f'q."{n}"' for n in QA_NAMES)
                + " FROM keys k JOIN read_parquet(?) f USING(channel_id,prediction_time) "
                "LEFT JOIN corrections c USING(channel_id,prediction_time) "
                "JOIN qa_delta q USING(channel_id,prediction_time)",
                [str(feature_source)],
            )
            masks = ",".join(
                f'CAST("{n}" IS NULL AS TINYINT) AS "missing__{n}"'
                for n in base_names
                if n != "sensor_type"
            )
            feature_path = folder / "new_model_features.parquet"
            db.execute(
                "COPY (SELECT *,"
                + masks
                + " FROM feature_delta ORDER BY channel_id,prediction_time) TO "
                + quoted(str(feature_path))
                + " (FORMAT PARQUET,COMPRESSION ZSTD)"
            )
            expected = db.execute("SELECT COUNT(*) FROM keys").fetchone()[0]
            actual, unique, mismatches = db.execute(
                "SELECT COUNT(*),COUNT(DISTINCT(f.channel_id,f.prediction_time)),"
                "COUNT(*) FILTER(WHERE f.sensor_type IS DISTINCT FROM a.sensor_type) "
                "FROM read_parquet(?) f JOIN candidates a USING(channel_id,prediction_time)",
                [str(feature_path)],
            ).fetchone()
            if (
                actual != expected
                or unique != expected
                or mismatches
                or (
                    pq.read_schema(feature_path).names
                    != ["channel_id", "prediction_time", *feature_names]
                )
            ):
                raise ValueError("delta feature key/type/schema mismatch")
            metadata = {
                "month": label,
                "decision_rows_checked": count,
                "new_feature_rows": expected,
                "groups": groups,
                "raw_quality_count_mismatches": 0,
                "files": {
                    p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                    for p in (changed, feature_path)
                },
                "source_admission_sha256": sha256(admission),
                "source_a3_features_sha256": a["features_sha256"],
            }
            write_json(folder / "manifest.json", metadata)
            months.append(
                {
                    **metadata,
                    "manifest_file": (folder / "manifest.json").relative_to(pending).as_posix(),
                    "manifest_sha256": sha256(folder / "manifest.json"),
                }
            )
            print(
                {
                    "month": label,
                    "checked": count,
                    "new_feature_rows": expected,
                    "seconds": round(time.perf_counter() - begun, 1),
                },
                flush=True,
            )
            for table in ("candidates", "evidence", "keys", "qa_delta", "feature_delta"):
                db.execute(f"DROP TABLE {table}")
        # Complete all causal artifacts before reading future outcomes.
        selected = {}
        for row in reversed(samples):
            key = row["sensor_type"], row["can_cold_start"], row["can_reenter"]
            selected.setdefault(key, row)
        replay_samples = list(selected.values())[:20]
        # The problematic channel was identified in the prior frozen audit;
        # select its first/last changed hours without labels or model scores.
        protected = (
            db.execute(
                "SELECT * FROM read_parquet(?,hive_partitioning=false) WHERE channel_id='228571' "
                "QUALIFY ROW_NUMBER() OVER(ORDER BY prediction_time)=1 OR "
                "ROW_NUMBER() OVER(ORDER BY prediction_time DESC)=1",
                [
                    [
                        str(pending / Path(m["manifest_file"]).parent / "new_admission.parquet")
                        for m in months
                    ]
                ],
            )
            .to_arrow_table()
            .to_pylist()
        )
        replay_samples = list(
            {
                (r["channel_id"], r["prediction_time"]): r for r in [*replay_samples, *protected]
            }.values()
        )
        replay = verify_replay(db, replay_samples, files)
        write_json(pending / "raw_replay_verification.json", replay)
        positive = pq.ParquetFile(q2_dir / "positive_hour_diagnostics.parquet").read().to_pylist()
        db.register("positive", pa.Table.from_pylist(positive))
        db.execute("CREATE TEMP TABLE positive_materialized AS SELECT * FROM positive")
        db.execute(
            "CREATE OR REPLACE TEMP VIEW admission AS SELECT "
            + ",".join(PAST_FIELDS)
            + ",CASE WHEN year(prediction_time)<2021 THEN 0 ELSE 1 END AS archive_segment FROM positive_materialized"
        )
        db.execute("CREATE OR REPLACE TEMP TABLE evidence AS " + evidence_sql())
        if db.execute(
            "SELECT COUNT(*) FROM evidence WHERE quality_rows_24h<>excluded_quality_count_24h"
        ).fetchone()[0]:
            raise ValueError("positive raw evidence mismatch")
        db.execute("CREATE OR REPLACE TEMP TABLE diagnostic_candidates AS " + policy_sql())
        columns = ",".join(f"d.{p}_status,d.{p}_reasons" for p in POLICIES)
        db.execute(
            "CREATE TEMP TABLE positive_result AS SELECT l.*,"
            + columns
            + " FROM positive l JOIN diagnostic_candidates d USING(channel_id,prediction_time)"
        )
        diagnostic = db.execute(
            "SELECT * FROM positive_result ORDER BY split,target_episode_id,prediction_time"
        ).to_arrow_table()
        pq.write_table(diagnostic, pending / "positive_hour_audit.parquet", compression="zstd")
        episodes, totals, by_type = summarize_episodes(diagnostic.to_pylist())
        pq.write_table(
            pa.Table.from_pylist(episodes), pending / "episode_coverage.parquet", compression="zstd"
        )
        for split in ("train", "validation"):
            original = report["label_audit"]["by_split"][split]
            if totals[split]["all_episodes"] != original["all_episodes"] or (
                totals[split]["base_available"] != original["candidate_available_episodes"]
            ):
                raise ValueError("full episode denominator or original coverage differs")
        # All changed hours, including negative/unknown/purged, not just positives.
        population = []
        for month in months:
            label = month["month"]
            b = b_chunks[label]
            bm = safe_path(b3_dir, b["manifest_file"], month=label)
            if sha256(bm) != b["manifest_sha256"]:
                raise ValueError("B3 month manifest differs")
            bp = bm.parent / "registered_forecast_labels.parquet"
            if sha256(bp) != read_json(bm)["files"][bp.name]["sha256"]:
                raise ValueError("immutable B3 labels differ")
            delta = pending / Path(month["manifest_file"]).parent / "new_admission.parquet"
            for policy in POLICIES:
                items = (
                    db.execute(
                        "SELECT l.target,l.label_status,l.split_status,COUNT(*) AS hours "
                        "FROM read_parquet(?,hive_partitioning=false) d JOIN read_parquet(?,hive_partitioning=false) l "
                        "USING(channel_id,prediction_time) WHERE d."
                        + policy
                        + "_status='eligible' GROUP BY ALL",
                        [str(delta), str(bp)],
                    )
                    .to_arrow_table()
                    .to_pylist()
                )
                for item in items:
                    population.append({"month": label, "policy": policy, **item})
                if (
                    policy == "combined"
                    and sum(r["hours"] for r in items) != month["new_feature_rows"]
                ):
                    raise ValueError("delta labels missing/duplicated; unknown cannot be dropped")
            label_proofs.append({"month": label, "sha256": sha256(bp)})
    total_rows = sum(m["decision_rows_checked"] for m in months)
    result = {
        "schema_version": VERSION,
        "status": "research_coverage_delta_complete_b_review_required",
        "policies": list(POLICIES),
        "source_q2_manifest_sha256": Q2_PIN,
        "source_manifests": pins,
        "source_m1_files": report["source_m1_files"],
        "source_b3_labels": label_proofs,
        "months": months,
        "all_decisions_checked": total_rows,
        "new_feature_rows": sum(m["new_feature_rows"] for m in months),
        "episode_coverage": totals,
        "by_type": by_type,
        "new_hour_label_population": population,
        "source_feature_allowlist_sha256": manifest["files"]["model_feature_allowlist.json"][
            "sha256"
        ],
        "feature_count": len(feature_names),
        "raw_replay": replay,
        "labels_read_for_causal_phase": False,
        "labels_changed": False,
        "production_gate_changed": False,
        "frozen_r6_changed": False,
        "test_events_read": False,
        "training_ready": False,
        "physical_availability_status": "unknown",
        "physical_failure_claim": False,
        "thresholds_or_models_trained": False,
        "new_model_metrics_computed": False,
        "limitations": [
            "After-normal reentry is an unapproved state checkpoint proposal, not physical availability or repair.",
            "Old conflicting measurements remain excluded; no same-second ordering or numeric/text compatibility is inferred.",
            "Model input is the union of unchanged Q2 eligible features and this delta filtered by declared policy.",
            "Unknown future labels remain unknown; this delta is not an automatically approved training sample.",
            "Coverage is diagnostic on seen train/validation; no performance guarantee or new independent test.",
        ],
        "code_lf_sha256": {p: frozen_rule_sha256(Path(p)) for p in CODE},
        "resources": {
            "elapsed_seconds": round(time.perf_counter() - begun, 3),
            "peak_working_set_bytes": getattr(psutil.Process().memory_info(), "peak_wset", 0),
            "duckdb_threads": 1,
            "duckdb_memory_limit": "1500MB",
            "reused_causal_months": len(months) if resume_from is not None else 0,
            "timing_scope": "publication_recovery_and_reverification"
            if resume_from is not None
            else "cold_build",
        },
    }
    write_json(pending / "report.json", result)
    write_json(
        pending / "manifest.json",
        {
            "schema_version": VERSION,
            "training_ready": False,
            "source_q2_manifest_sha256": Q2_PIN,
            "months": months,
            "files": {
                n: {"sha256": sha256(pending / n), "bytes": (pending / n).stat().st_size}
                for n in (
                    "report.json",
                    "positive_hour_audit.parquet",
                    "episode_coverage.parquet",
                    "raw_replay_verification.json",
                )
            },
        },
    )
    pending.rename(output_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("q2-dir", "m1-dir", "a3-dir", "b3-dir", "corrections-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    result = build(**vars(args))
    print(
        {
            k: result[k]
            for k in ("all_decisions_checked", "new_feature_rows", "episode_coverage", "resources")
        }
    )


if __name__ == "__main__":
    main()
