"""Independent B acceptance checks for A's Q3 admission delta, without training."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from analysis.verify_q2_oracle_b import channel_capacity


POLICIES = ("cold_start", "after_normal", "combined")
EXPECTED_MONTHS = [f"{year}-{month:02d}" for year in (2019, 2020, 2022, 2023, 2024, 2025)
                   for month in range(1, 13)]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _quoted(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _check_new_rows(db: duckdb.DuckDBPyConnection, delta: Path,
                    old: Path, features: Path) -> tuple[int, dict]:
    db.execute("CREATE OR REPLACE TEMP VIEW d AS SELECT * FROM read_parquet("
               + _quoted(delta) + ",hive_partitioning=false)")
    db.execute("CREATE OR REPLACE TEMP VIEW f AS SELECT * FROM read_parquet("
               + _quoted(features) + ",hive_partitioning=false)")
    db.execute("CREATE OR REPLACE TEMP VIEW old AS SELECT channel_id,prediction_time,"
               "sensor_type,admission_status,admission_reasons,last_explicit_normal_at "
               "FROM read_parquet(" + _quoted(old) + ",hive_partitioning=false)")
    count, unique, feature_count, feature_unique = db.execute("""
        SELECT (SELECT count(*) FROM d),
               (SELECT count(DISTINCT(channel_id,prediction_time)) FROM d),
               (SELECT count(*) FROM f),
               (SELECT count(DISTINCT(channel_id,prediction_time)) FROM f)
    """).fetchone()
    if count != unique or count != feature_count or count != feature_unique:
        raise ValueError("delta/feature key multiplicity differs")
    bad_join = db.execute("""SELECT count(*) FROM d
        FULL JOIN f USING(channel_id,prediction_time)
        LEFT JOIN old ON old.channel_id=coalesce(d.channel_id,f.channel_id)
                     AND old.prediction_time=coalesce(d.prediction_time,f.prediction_time)
        WHERE d.channel_id IS NULL OR f.channel_id IS NULL OR old.channel_id IS NULL
           OR d.sensor_type IS DISTINCT FROM f.sensor_type
           OR d.sensor_type IS DISTINCT FROM old.sensor_type
           OR old.admission_status<>'unknown'
           OR d.admission_status IS DISTINCT FROM old.admission_status
           OR d.admission_reasons IS DISTINCT FROM old.admission_reasons
           OR d.last_explicit_normal_at IS DISTINCT FROM old.last_explicit_normal_at
    """).fetchone()[0]
    if bad_join:
        raise ValueError(f"{bad_join} new rows do not preserve old admission/feature keys")
    bad_guard = db.execute("""SELECT count(*) FROM d WHERE
        admission_status<>'unknown' OR combined_status<>'eligible'
        OR len(combined_reasons)<>0 OR availability_status<>'unknown'
        OR len(admission_reasons)=0
        OR len(list_filter(admission_reasons,
              r->r NOT IN ('insufficient_history','quality_exclusions_24h')))<>0
        OR cold_start_reasons IS DISTINCT FROM
           list_filter(admission_reasons,r->r<>'insufficient_history')
        OR after_normal_reasons IS DISTINCT FROM
           list_filter(admission_reasons,r->r<>'quality_exclusions_24h')
        OR cold_start_status IS DISTINCT FROM
           CASE WHEN list_contains(admission_reasons,'quality_exclusions_24h')
                THEN 'unknown' ELSE 'eligible' END
        OR after_normal_status IS DISTINCT FROM
           CASE WHEN list_contains(admission_reasons,'insufficient_history')
                THEN 'unknown' ELSE 'eligible' END
        OR blocking_qa_count_24h<>0 OR ambiguous_seconds_24h<>0
        OR last_explicit_normal_at IS NULL OR last_explicit_normal_at>prediction_time
        OR last_explicit_normal_at<prediction_time-INTERVAL '168 hours'
        OR admission_evidence_through IS NULL OR admission_evidence_through>prediction_time
        OR quality_rows_24h<>excluded_quality_count_24h
        OR (list_contains(admission_reasons,'insufficient_history') AND
            (first_usable_at IS NULL OR second_usable_at IS NULL
             OR second_usable_at<=first_usable_at OR second_usable_at>prediction_time))
        OR (list_contains(admission_reasons,'quality_exclusions_24h') AND
            (quality_rows_24h<=0 OR last_conflict_at IS NULL
             OR last_conflict_at<=prediction_time-INTERVAL '24 hours'
             OR last_conflict_at>=last_explicit_normal_at
             OR last_hard_quality_at>prediction_time-INTERVAL '24 hours'))
        OR year(prediction_time) IN (2021,2026)
    """).fetchone()[0]
    if bad_guard:
        raise ValueError(f"{bad_guard} new rows violate independent guard checks")
    groups = db.execute("""SELECT
        list_contains(admission_reasons,'insufficient_history') AS needs_cold,
        list_contains(admission_reasons,'quality_exclusions_24h') AS needs_normal,
        count(*) AS hours FROM d GROUP BY 1,2 ORDER BY 1,2""").fetchall()
    return count, {f"cold={cold},normal={normal}": n for cold, normal, n in groups}


def _oracle(points: pd.DataFrame, status: str) -> tuple[int, int]:
    selected = points.loc[(points.split == "validation") & (points[status] == "eligible")]
    available = selected.target_episode_id.nunique()
    if not selected.groupby("target_episode_id").prediction_time.agg(
        lambda times: times.max() - times.min()
    ).lt(pd.Timedelta(hours=24)).all():
        raise ValueError("positive hours of an episode exceed cooldown")
    maximum = 0
    for _, group in selected.groupby("channel_id"):
        optimal, greedy = channel_capacity(sorted(group.prediction_time.tolist()))
        if optimal != greedy:
            raise ValueError("B greedy and backward DP disagree")
        maximum += optimal
    return available, maximum


def review(*, package: Path, q2_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest = _json(package / "manifest.json")
    report = _json(package / "report.json")
    if (manifest["schema_version"] != "q3-a-coverage-reentry-research-v1"
            or manifest["training_ready"] or report["training_ready"]
            or report["labels_changed"] or report["frozen_r6_changed"]
            or report["production_gate_changed"] or report["test_events_read"]
            or report["thresholds_or_models_trained"]
            or manifest["source_q2_manifest_sha256"] != _hash(q2_dir / "manifest.json")
            or [item["month"] for item in manifest["months"]] != EXPECTED_MONTHS):
        raise ValueError("unaccepted Q3 source, readiness or month scope")
    for name, item in manifest["files"].items():
        if _hash(package / name) != item["sha256"]:
            raise ValueError(f"Q3 root file changed: {name}")
    base_schema = pq.read_schema(q2_dir / "year=2025/month=12/model_features.parquet")
    rows = 0
    by_reason: Counter[str] = Counter()
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=1")
        db.execute("SET memory_limit='1500MB'")
        for item in manifest["months"]:
            folder = package / Path(item["manifest_file"]).parent
            if _hash(folder / "manifest.json") != item["manifest_sha256"]:
                raise ValueError(f"Q3 month manifest changed: {item['month']}")
            for name, entry in item["files"].items():
                if _hash(folder / name) != entry["sha256"]:
                    raise ValueError(f"Q3 payload changed: {item['month']}/{name}")
            delta = folder / "new_admission.parquet"
            features = folder / "new_model_features.parquet"
            if not pq.read_schema(features).equals(base_schema):
                raise ValueError(f"Q3 feature schema changed: {item['month']}")
            old = q2_dir / Path(item["manifest_file"]).parent / "admission.parquet"
            if _hash(old) != item["source_admission_sha256"]:
                raise ValueError(f"Q2 source admission changed: {item['month']}")
            count, groups = _check_new_rows(db, delta, old, features)
            if count != item["new_feature_rows"]:
                raise ValueError(f"Q3 count differs in {item['month']}")
            rows += count
            by_reason.update(groups)
    audit = pq.read_table(package / "positive_hour_audit.parquet").to_pandas()
    old_audit = pq.read_table(q2_dir / "positive_hour_diagnostics.parquet").to_pandas()
    original_fields = old_audit.columns.tolist()
    keys = ["channel_id", "prediction_time"]
    if not audit.sort_values(keys).reset_index(drop=True)[original_fields].equals(
        old_audit.sort_values(keys).reset_index(drop=True)
    ):
        raise ValueError("positive-hour labels or original fields changed")
    full = pq.read_table(package / "episode_coverage.parquet").to_pandas()
    validation = full.loc[full.split == "validation"]
    if (validation.target_episode_id.nunique() != len(validation)
            or len(validation) != 2142):
        raise ValueError("full episode denominator changed")
    remaining = validation.loc[validation.coverage_category == "still_protected"]
    remaining_reasons = Counter(
        tuple(value) for value in remaining.remaining_reasons_every_hour
    )
    cold_hours = audit.loc[(audit.split == "validation")
                           & audit.admission_status.ne("eligible")
                           & audit.cold_start_status.eq("eligible")]
    cold_second_before_normal = cold_hours.loc[
        cold_hours.second_usable_at < cold_hours.last_explicit_normal_at
    ]
    capacities = {}
    for policy, status in (("base", "admission_status"),
                           *((name, f"{name}_status") for name in POLICIES)):
        available, maximum = _oracle(audit, status)
        coverage_field = "base_eligible_hours" if policy == "base" else f"{policy}_eligible_hours"
        if available != validation.loc[validation[coverage_field] > 0].shape[0]:
            raise ValueError(f"{policy} episode coverage differs from hourly audit")
        capacities[policy] = {"available_episodes": available,
                              "maximum_24h_oracle_matches": maximum,
                              "full_recall_oracle": maximum / len(validation)}
    result = {
        "schema_version": "q3-b-independent-coverage-review-v1",
        "source_package_manifest_sha256": _hash(package / "manifest.json"),
        "source_q2_manifest_sha256": _hash(q2_dir / "manifest.json"),
        "months_verified": len(manifest["months"]),
        "new_rows_verified": rows,
        "new_rows_by_relaxed_reason": dict(by_reason),
        "validation_episodes": len(validation),
        "validation_coverage_categories": {
            str(key): int(value) for key, value in validation.coverage_category.value_counts().items()
        },
        "remaining_19_reason_counts": {"+".join(key): count for key, count in
                                       remaining_reasons.items()},
        "new_cold_start_positive_hours": len(cold_hours),
        "new_cold_start_positive_hours_with_second_observation_before_normal": len(
            cold_second_before_normal
        ),
        "new_cold_start_episodes_with_second_observation_before_normal": int(
            cold_second_before_normal.target_episode_id.nunique()
        ),
        "oracle": capacities,
        "model_trained_or_threshold_selected": False,
        "future_labels_used_only_for_retrospective_coverage_and_oracle": True,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "q2-dir", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    result = review(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in (
        "months_verified", "new_rows_verified", "validation_coverage_categories", "oracle"
    )}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
