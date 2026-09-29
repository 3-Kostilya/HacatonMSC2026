"""Independent full delta guard/feature checks and exact 24h oracle capacities."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time

import duckdb
import pyarrow.parquet as pq

from analysis.audit_q2_oracle_a import optimal_schedule
from analysis.build_quality_improvement_a import QA_NAMES, expected_months, quoted
from analysis.build_sparse_population_a import write_json
from analysis.coverage_reentry_a import PAST_FIELDS, POLICIES
from analysis.train_r4_discrete_baselines import read_json, sha256


def guard_violations_sql():
    """Independent invariants on NEW rows, not reuse of candidate policy SQL."""
    return """SELECT COUNT(*) FROM delta WHERE
        admission_status<>'unknown' OR combined_status<>'eligible' OR len(combined_reasons)>0
        OR NOT list_has_any(admission_reasons,['insufficient_history','quality_exclusions_24h'])
        OR len(list_filter(admission_reasons,r->r NOT IN ('insufficient_history','quality_exclusions_24h')))>0
        OR availability_status<>'unknown' OR blocking_qa_count_24h<>0 OR ambiguous_seconds_24h<>0
        OR last_explicit_normal_at IS NULL OR last_explicit_normal_at>prediction_time
        OR last_explicit_normal_at<prediction_time-INTERVAL '168 hours'
        OR admission_evidence_through IS NULL OR admission_evidence_through>prediction_time
        OR quality_rows_24h<>excluded_quality_count_24h
        OR (list_contains(admission_reasons,'insufficient_history') AND
            (first_usable_at IS NULL OR second_usable_at IS NULL OR second_usable_at<=first_usable_at
             OR second_usable_at>prediction_time))
        OR (list_contains(admission_reasons,'quality_exclusions_24h') AND
            (last_conflict_at IS NULL OR last_conflict_at<=prediction_time-INTERVAL '24 hours'
             OR last_conflict_at>=last_explicit_normal_at OR quality_rows_24h<=0
             OR last_hard_quality_at>prediction_time-INTERVAL '24 hours'))
        OR cold_start_reasons IS DISTINCT FROM list_filter(admission_reasons,r->r<>'insufficient_history')
        OR after_normal_reasons IS DISTINCT FROM list_filter(admission_reasons,r->r<>'quality_exclusions_24h')
        OR cold_start_status IS DISTINCT FROM CASE WHEN len(cold_start_reasons)=0 THEN 'eligible' ELSE 'unknown' END
        OR after_normal_status IS DISTINCT FROM CASE WHEN len(after_normal_reasons)=0 THEN 'eligible' ELSE 'unknown' END
        OR year(prediction_time) IN (2021,2026)
    """


def verify(*, package, q2_dir, a3_dir, corrections_dir, output):
    begun = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    manifest, report = [read_json(package / n) for n in ("manifest.json", "report.json")]
    if sha256(q2_dir / "manifest.json") != manifest["source_q2_manifest_sha256"]:
        raise ValueError("original Q2 reference differs")
    for name, info in manifest["files"].items():
        if sha256(package / name) != info["sha256"]:
            raise ValueError("coverage report/diagnostic differs")
    if [m["month"] for m in manifest["months"]] != expected_months():
        raise ValueError("incomplete coverage months")
    if sha256(a3_dir / "manifest.json") != report["source_manifests"]["a3"]:
        raise ValueError("A3 provenance differs")
    correction = corrections_dir / "feature_corrections.parquet"
    if sha256(correction) != report["source_manifests"]["corrections"]:
        raise ValueError("QA correction source differs")
    base = read_json(q2_dir / "model_feature_allowlist.json")["base_feature_names"]
    ac = {m["month"]: m for m in read_json(a3_dir / "manifest.json")["chunks"]}
    rows, checked_months = 0, []
    with duckdb.connect() as db:
        db.execute("SET threads=1")
        db.execute("SET memory_limit='1500MB'")
        db.execute(
            "CREATE TEMP VIEW corrections AS SELECT * FROM read_parquet("
            + quoted(str(correction))
            + ")"
        )
        for month in manifest["months"]:
            label = month["month"]
            folder = package / Path(month["manifest_file"]).parent
            if sha256(folder / "manifest.json") != month["manifest_sha256"]:
                raise ValueError("month manifest differs")
            for name, info in month["files"].items():
                if sha256(folder / name) != info["sha256"]:
                    raise ValueError("month payload differs")
            delta, features = (
                folder / "new_admission.parquet",
                folder / "new_model_features.parquet",
            )
            db.execute(
                "CREATE OR REPLACE TEMP VIEW delta AS SELECT * FROM read_parquet("
                + quoted(str(delta))
                + ")"
            )
            db.execute(
                "CREATE OR REPLACE TEMP VIEW features AS SELECT * FROM read_parquet("
                + quoted(str(features))
                + ")"
            )
            if db.execute(guard_violations_sql()).fetchone()[0]:
                raise ValueError("independent guard invariants failed")
            count, unique = db.execute(
                "SELECT COUNT(*),COUNT(DISTINCT(channel_id,prediction_time)) FROM delta"
            ).fetchone()
            if count != unique or count != month["new_feature_rows"]:
                raise ValueError("changed key count differs")
            original = q2_dir / Path(month["manifest_file"]).parent / "admission.parquet"
            if sha256(original) != month["source_admission_sha256"]:
                raise ValueError("original admission differs")
            comparisons = " OR ".join(f'd."{n}" IS DISTINCT FROM a."{n}"' for n in PAST_FIELDS)
            mismatch = db.execute(
                "SELECT COUNT(*) FROM delta d LEFT JOIN read_parquet(?) a "
                "USING(channel_id,prediction_time) WHERE a.channel_id IS NULL OR " + comparisons,
                [str(original)],
            ).fetchone()[0]
            if mismatch:
                raise ValueError("certified original fields were rewritten")
            count_f, unique_f = db.execute(
                "SELECT COUNT(*),COUNT(DISTINCT(channel_id,prediction_time)) FROM features"
            ).fetchone()
            if (count_f, unique_f) != (count, count) or db.execute(
                "SELECT COUNT(*) FROM delta d FULL OUTER JOIN features f USING(channel_id,prediction_time) "
                "WHERE d.channel_id IS NULL OR f.channel_id IS NULL OR d.sensor_type IS DISTINCT FROM f.sensor_type"
            ).fetchone()[0]:
                raise ValueError("complete feature keys/types differ")
            masks = " OR ".join(
                f'f."missing__{n}" IS DISTINCT FROM CAST(f."{n}" IS NULL AS TINYINT)'
                for n in base
                if n != "sensor_type"
            )
            if db.execute("SELECT COUNT(*) FROM features f WHERE " + masks).fetchone()[0]:
                raise ValueError("missing statistics were not correctly flagged")
            source = a3_dir / ac[label]["features_file"]
            if sha256(source) != month["source_a3_features_sha256"]:
                raise ValueError("original A3 features differ")
            comparisons = " OR ".join(
                f'f."{n}" IS DISTINCT FROM CASE WHEN c.channel_id IS NOT NULL THEN c."{n}" ELSE a."{n}" END'
                for n in base
            )
            if db.execute(
                "SELECT COUNT(*) FROM features f LEFT JOIN read_parquet(?) a "
                "USING(channel_id,prediction_time) LEFT JOIN corrections c USING(channel_id,prediction_time) "
                "WHERE a.channel_id IS NULL OR " + comparisons,
                [str(source)],
            ).fetchone()[0]:
                raise ValueError("base features differ from approved past calculations/corrections")
            if db.execute(
                "SELECT COUNT(*) FROM features WHERE "
                + " OR ".join(f'"{n}" IS NULL OR "{n}"<0' for n in QA_NAMES)
            ).fetchone()[0]:
                raise ValueError("invalid QA count")
            rows += count
            checked_months.append(
                {
                    "month": label,
                    "rows": count,
                    "guard_violations": 0,
                    "base_feature_mismatches": 0,
                    "key_mismatches": 0,
                }
            )
    positive = pq.ParquetFile(package / "positive_hour_audit.parquet").read().to_pandas()
    original_positive = (
        pq.ParquetFile(q2_dir / "positive_hour_diagnostics.parquet").read().to_pandas()
    )
    original_names = original_positive.columns.tolist()
    keys = ["channel_id", "prediction_time"]
    left = positive.sort_values(keys).reset_index(drop=True)[original_names]
    right = original_positive.sort_values(keys).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("positive labels/original causal fields changed")
    episode_rows = pq.ParquetFile(package / "episode_coverage.parquet").read().to_pylist()
    categories = Counter(r["coverage_category"] for r in episode_rows if r["split"] == "validation")
    if sum(categories.values()) != 2142 or categories["already_available"] != 1359:
        raise ValueError("episode partition is not all 2142 episodes")
    oracle = {}
    val = positive.loc[positive.split == "validation"]
    for policy, field in (("base", "admission_status"), *[(p, p + "_status") for p in POLICIES]):
        points = val.loc[val[field] == "eligible"]
        witness, channels = optimal_schedule(points)
        oracle[policy] = {
            "available_episodes": int(points.target_episode_id.nunique()),
            "maximum_matched_at_24h_cooldown": len(witness),
            "full_recall_capacity_at_24h_cooldown": len(witness) / 2142,
            "greedy_dp_agreement": all(
                c["greedy_matches"] == c["dynamic_program_matches"] for c in channels
            ),
        }
    if (
        rows != report["new_feature_rows"]
        or oracle["base"]["maximum_matched_at_24h_cooldown"] != 1116
    ):
        raise ValueError("full delta total/old oracle differs")
    result = {
        "status": "independent_delta_invariants_features_and_oracle_verified",
        "source_manifest_sha256": sha256(package / "manifest.json"),
        "months": checked_months,
        "new_rows_verified": rows,
        "positive_rows_unchanged": len(positive),
        "validation_episode_partition": dict(categories),
        "oracle": oracle,
        "oracle_uses_future_labels_not_model_quality": True,
        "production_or_joint_approval": False,
        "elapsed_seconds": round(time.perf_counter() - begun, 3),
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "q2-dir", "a3-dir", "corrections-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = verify(**vars(parser.parse_args()))
    print(
        {
            k: result[k]
            for k in (
                "new_rows_verified",
                "validation_episode_partition",
                "oracle",
                "elapsed_seconds",
            )
        }
    )


if __name__ == "__main__":
    main()
