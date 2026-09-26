"""Full train/validation QA features and past-data coverage audit; never read test data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

import duckdb
import psutil

from analysis.build_a2_hourly import _monthly_files
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import read_json, sha256, sha256_pinned_text
from stage1.features.hourly import FeatureEvent
from stage1.features.qa_values import QA_CATEGORIES, qa_window_counts
from stage1.features.schema import WINDOW_HOURS
from stage1.value_quality import (
    EPOCH_VALUE_ARTIFACTS,
    QA_VALUE_RULESET_VERSION,
    TEMPERATURE_SERVICE_CODE_CANDIDATES,
)


VERSION = "quality-improvement-a-qa-features-v1"
QA_NAMES = tuple(
    f"qa_{category}_count_{hours}h" for hours in WINDOW_HOURS for category in QA_CATEGORIES
)
KEYS = ("channel_id", "prediction_time")
END = datetime(2026, 1, 1)
TRAIN_END = datetime(2025, 1, 1)
YEARS = (2019, 2020, 2022, 2023, 2024, 2025)


def expected_months() -> list[str]:
    return [f"{year}-{month:02d}" for year in YEARS for month in range(1, 13)]


def quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def safe_path(root: Path, relative: str, *, month: str | None = None) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("input manifest path escapes its root")
    if month is not None:
        parent = root / f"year={month[:4]}" / f"month={month[5:]}"
        if path.parent != parent.resolve():
            raise ValueError("input manifest points outside the permitted month")
    return path


def category_sql() -> str:
    epoch = ",".join(quoted(value) for value in sorted(EPOCH_VALUE_ARTIFACTS))
    temperatures = ",".join(str(value) for value in sorted(TEMPERATURE_SERVICE_CODE_CANDIDATES))
    spaces = "".join(
        chr(code)
        for code in (
            *range(9, 14),
            *range(28, 33),
            133,
            160,
            5760,
            *range(8192, 8203),
            8232,
            8233,
            8239,
            8287,
            12288,
        )
    )
    raw = f"trim(COALESCE(NULLIF(value_raw,''),value_state,''),{quoted(spaces)})"
    return f"""CASE
        WHEN value_numeric IS NULL AND {raw} IN ({epoch}) THEN 'epoch_value_artifact'
        WHEN isfinite(value_numeric) AND sensor_type='Датчик температуры'
             AND value_numeric IN ({temperatures}) THEN 'temperature_service_code_candidate'
        WHEN isfinite(value_numeric) AND sensor_type='Газовый датчик'
             AND value_numeric<0 THEN 'gas_negative_reading'
        WHEN isfinite(value_numeric) AND sensor_type='Газовый датчик'
             AND value_numeric>100 THEN 'gas_above_physical_percent'
        WHEN isfinite(value_numeric) AND sensor_type='Газовый датчик'
             AND value_numeric>=1 THEN 'gas_alarm_level_candidate'
        ELSE NULL END"""


def create_qa_prefixes(database, files: list[Path]) -> dict:
    # Same timestamp messages are aggregated before prefix sums, never ordered by row IDs.
    pieces = [
        f"SUM(CAST(category={quoted(category)} AS BIGINT)) AS n{i}"
        for i, category in enumerate(QA_CATEGORIES)
    ]
    sums = [
        f"CAST(SUM(n{i}) OVER (PARTITION BY channel_id,archive_segment ORDER BY timestamp "
        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS BIGINT) AS c{i}"
        for i in range(len(QA_CATEGORIES))
    ]
    database.execute(
        "CREATE TEMP TABLE qa_events AS SELECT channel_id,timestamp,category,"
        "CASE WHEN year(timestamp)<2021 THEN 0 ELSE 1 END AS archive_segment FROM ("
        "SELECT channel_id,timestamp,source," + category_sql() + " AS category "
        "FROM read_parquet(?,hive_partitioning=false)) WHERE category IS NOT NULL "
        "AND timestamp>=TIMESTAMP '2019-01-01' AND timestamp<TIMESTAMP '2026-01-01' "
        "AND year(timestamp)<>2021 AND split_part(replace(source,chr(92),'/'),'/',-1)="
        "'ext-journal-' || CAST(year(timestamp) AS VARCHAR) || '.7z'",
        [[str(path) for path in files]],
    )
    counts = database.execute(
        "SELECT category,COUNT(*) FROM qa_events GROUP BY category"
    ).fetchall()
    database.execute(
        "CREATE TEMP TABLE qa_prefix AS SELECT channel_id,timestamp,archive_segment,"
        + ",".join(sums)
        + " FROM (SELECT channel_id,timestamp,archive_segment,"
        + ",".join(pieces)
        + " FROM qa_events GROUP BY channel_id,timestamp,archive_segment)"
    )
    return dict(counts)


def counts_sql() -> str:
    joins = [
        "ASOF LEFT JOIN qa_prefix AS high ON k.channel_id=high.channel_id "
        "AND k.archive_segment=high.archive_segment AND k.prediction_time>=high.timestamp"
    ]
    columns = []
    for hours in WINDOW_HOURS:
        low = f"low{hours}"
        joins.append(
            f"ASOF LEFT JOIN qa_prefix AS {low} ON k.channel_id={low}.channel_id "
            f"AND k.archive_segment={low}.archive_segment "
            f"AND k.prediction_time-INTERVAL '{hours} hours'>={low}.timestamp"
        )
        for i, category in enumerate(QA_CATEGORIES):
            columns.append(
                f"CAST(COALESCE(high.c{i},0)-COALESCE({low}.c{i},0) AS BIGINT) "
                f'AS "qa_{category}_count_{hours}h"'
            )
    return (
        "SELECT k.channel_id,k.prediction_time,"
        + ",".join(columns)
        + " FROM keys k "
        + " ".join(joins)
    )


def coverage_month(database, features: Path, statuses: Path, keys: Path, expected: int) -> dict:
    database.execute(
        "CREATE OR REPLACE TEMP TABLE past_audit AS "
        "SELECT f.sensor_type,s.discrete_data_status,s.numeric_data_status,"
        "s.discrete_data_reasons,s.numeric_data_reasons,s.baseline_reasons,"
        "k.channel_id IS NOT NULL AS conditional_key "
        "FROM read_parquet(?) s JOIN read_parquet(?) f USING(channel_id,prediction_time) "
        "LEFT JOIN (SELECT channel_id,prediction_time FROM read_parquet(?)) k "
        "USING(channel_id,prediction_time)",
        [str(statuses), str(features), str(keys)],
    )
    count = database.execute("SELECT COUNT(*) FROM past_audit").fetchone()[0]
    if count != expected:
        raise ValueError("A3 coverage join has missing or duplicate keys")
    grouped = database.execute(
        "SELECT COALESCE(sensor_type,'<unknown_or_conflicting>'),discrete_data_status,"
        "numeric_data_status,conditional_key,COUNT(*) FROM past_audit GROUP BY ALL"
    ).fetchall()
    reasons = {}
    for field in ("discrete_data_reasons", "numeric_data_reasons", "baseline_reasons"):
        # Each reason counts affected rows, not independent observations or mutually exclusive causes.
        reasons[field] = dict(
            database.execute(
                f"SELECT reason,COUNT(*) FROM past_audit,UNNEST({field}) AS r(reason) GROUP BY reason"
            ).fetchall()
        )
    return {"rows": count, "groups": [list(row) for row in grouped], "reasons": reasons}


def verify_python_samples(database, qa_file: Path, m1_files: list[Path]) -> list[dict]:
    # Nonzero cases plus a deterministic sample; labels and test events are never read.
    cases = database.execute(
        "SELECT channel_id,prediction_time FROM read_parquet(?) ORDER BY "
        "sha256(channel_id || CAST(prediction_time AS VARCHAR)) LIMIT 2",
        [str(qa_file)],
    ).fetchall()
    for name in QA_NAMES:
        case = database.execute(
            f'SELECT channel_id,prediction_time FROM read_parquet(?) WHERE "{name}">0 '
            "ORDER BY channel_id,prediction_time LIMIT 1",
            [str(qa_file)],
        ).fetchone()
        if case is not None:
            cases.append(case)
    checks = []
    for channel, at in sorted(set(cases)):
        query_lower = at - timedelta(hours=168)
        lower = max(query_lower, datetime(2022 if at.year >= 2022 else 2019, 1, 1))
        files = [
            path
            for path in m1_files
            if lower.replace(day=1, hour=0)
            <= datetime(
                int(path.parent.parent.name.split("=")[1]), int(path.parent.name.split("=")[1]), 1
            )
            <= at.replace(day=1, hour=0)
        ]
        records = (
            database.execute(
                "SELECT channel_id,timestamp,alarm,value_numeric,value_state,value_raw,sensor_type,"
                "quality_flags,source FROM read_parquet(?,hive_partitioning=false) "
                "WHERE channel_id=? AND timestamp>? AND timestamp<=? "
                "AND split_part(replace(source,chr(92),'/'),'/',-1)="
                "'ext-journal-' || CAST(year(timestamp) AS VARCHAR) || '.7z' ORDER BY timestamp,row_id",
                [[str(path) for path in files], channel, query_lower, at],
            )
            .to_arrow_table()
            .to_pylist()
        )
        expected = qa_window_counts(
            [
                FeatureEvent.from_clean_record(record, apply_qa_value_policy=True)
                for record in records
            ],
            at,
        )
        actual = (
            database.execute(
                "SELECT * FROM read_parquet(?) WHERE channel_id=? AND prediction_time=?",
                [str(qa_file), channel, at],
            )
            .to_arrow_table()
            .to_pylist()[0]
        )
        if any(actual[name] != expected[name] for name in QA_NAMES):
            raise ValueError(f"independent QA prefix check differs for {channel}@{at}")
        checks.append(
            {
                "channel_id": channel,
                "prediction_time": at.isoformat(),
                "source_rows": len(records),
                "mismatches": 0,
            }
        )
    return checks


def build(
    *,
    m1_dir: Path,
    a3_dir: Path,
    admission_dir: Path,
    corrections_dir: Path,
    output_dir: Path,
    contract_path: Path,
    allowlist_path: Path,
) -> dict:
    started = time.perf_counter()
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or pending.exists():
        raise FileExistsError(output_dir)
    contract, base = read_json(contract_path), read_json(allowlist_path)
    pins = (
        (m1_dir / "manifest.json", "source_m1_manifest_sha256"),
        (a3_dir / "manifest.json", "source_a3_manifest_sha256"),
        (admission_dir / "manifest.json", "source_admission_manifest_sha256"),
        (corrections_dir / "feature_corrections.parquet", "source_qa_correction_features_sha256"),
    )
    if any(sha256(path) != contract[key] for path, key in pins):
        raise ValueError("Q1 input provenance differs")
    if (
        contract["schema_version"] != VERSION
        or contract["test_data_access_enabled"]
        or contract["qa_value_ruleset_version"] != QA_VALUE_RULESET_VERSION
        or sha256_pinned_text(allowlist_path) != contract["source_base_allowlist_git_crlf_sha256"]
        or len(base["feature_names"]) != 51
        or len(set(base["feature_names"])) != 51
        or contract["windows_hours"] != list(WINDOW_HOURS)
        or contract["expected_months"] != 72
        or contract["train_end_exclusive"] != TRAIN_END.isoformat()
        or contract["validation_end_exclusive"] != END.isoformat()
        or contract["excluded_year"] != 2021
    ):
        raise ValueError("Q1 feature contract differs")
    a3, admission = read_json(a3_dir / "manifest.json"), read_json(admission_dir / "manifest.json")
    correction_report = read_json(corrections_dir / "report.json")
    if (
        a3["status"] != "complete"
        or admission["status"] != "complete_conditional_candidates"
        or a3["source_m1_manifest_sha256"] != contract["source_m1_manifest_sha256"]
        or admission["source_a3_manifest_sha256"] != contract["source_a3_manifest_sha256"]
        or correction_report["source_a3_manifest_sha256"] != contract["source_a3_manifest_sha256"]
        or correction_report["source_m1_manifest_sha256"] != contract["source_m1_manifest_sha256"]
    ):
        raise ValueError("Q1 sources do not share accepted lineage")
    a_chunks, i_chunks = ({row["month"]: row for row in data["chunks"]} for data in (a3, admission))
    months = expected_months()
    if any(month not in a_chunks or month not in i_chunks for month in months):
        raise ValueError("train/validation source month absent")
    m1_files = []
    for year in YEARS:
        files, missing = _monthly_files(m1_dir, datetime(year, 1, 1), datetime(year + 1, 1, 1))
        if missing:
            raise ValueError(f"train/validation M1 files missing: {missing}")
        m1_files.extend(files)
    if any("year=2026" in path.parts or "year=2021" in path.parts for path in m1_files):
        raise ValueError("test or excluded-year source selected")
    pending.mkdir(parents=True)
    base_names = base["feature_names"]
    allowlist = {
        "schema_version": VERSION,
        "feature_names": [*base_names, *QA_NAMES],
        "base_feature_names": base_names,
        "qa_feature_names": list(QA_NAMES),
        "categorical_feature_names": ["sensor_type"],
        "feature_count": 71,
        "keys_not_features": list(KEYS),
        "diagnostics_and_labels_not_features": True,
        "training_ready": False,
        "independent_b_review_required": True,
    }
    (pending / "model_feature_allowlist.json").write_text(
        json.dumps(allowlist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    totals = defaultdict(Counter)
    reasons = defaultdict(lambda: defaultdict(Counter))
    by_type = defaultdict(lambda: defaultdict(Counter))
    support = defaultdict(Counter)
    month_reports, checks, source_files = [], [], []
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        database.execute("SET memory_limit='2GB'")
        database.execute("SET preserve_insertion_order=false")
        qa_events = create_qa_prefixes(database, m1_files)
        database.execute(
            "CREATE TEMP TABLE corrections AS SELECT * FROM read_parquet(?) "
            "WHERE prediction_time<TIMESTAMP '2026-01-01' AND year(prediction_time)<>2021",
            [str(corrections_dir / "feature_corrections.parquet")],
        )
        duplicates = database.execute(
            "SELECT COUNT(*) FROM (SELECT channel_id,prediction_time "
            "FROM corrections GROUP BY ALL HAVING COUNT(*)>1)"
        ).fetchone()[0]
        if duplicates:
            raise ValueError("duplicate correction key")
        for month in months:
            split = "validation" if month[:4] == "2025" else "train"
            a, index = a_chunks[month], i_chunks[month]
            a_manifest = safe_path(a3_dir, a["manifest_file"], month=month)
            i_manifest = safe_path(admission_dir, index["manifest_file"], month=month)
            if (
                sha256(a_manifest) != a["manifest_sha256"]
                or sha256(i_manifest) != index["manifest_sha256"]
            ):
                raise ValueError(f"source month manifest hash differs: {month}")
            im = read_json(i_manifest)
            features = safe_path(a3_dir, a["features_file"], month=month)
            statuses = safe_path(a3_dir, a["row_status_file"], month=month)
            candidate = i_manifest.parent / "conditional_discrete_keys.parquet"
            for path, expected in (
                (features, a["features_sha256"]),
                (statuses, a["row_status_sha256"]),
                (candidate, im["candidate_sha256"]),
            ):
                actual = sha256(path)
                if actual != expected:
                    raise ValueError(f"source file hash differs: {path}")
                source_files.append({"month": month, "file": path.name, "sha256": actual})
            database.execute(
                "CREATE OR REPLACE TEMP TABLE keys AS SELECT channel_id,prediction_time,"
                "sensor_type,CASE WHEN year(prediction_time)<2021 THEN 0 ELSE 1 END "
                "AS archive_segment FROM read_parquet(?)",
                [str(candidate)],
            )
            key_count, unique, low, high = database.execute(
                "SELECT COUNT(*),COUNT(DISTINCT (channel_id,prediction_time)),"
                "MIN(prediction_time),MAX(prediction_time) FROM keys"
            ).fetchone()
            if (
                key_count != im["row_count"]
                or unique != key_count
                or (
                    key_count
                    and (low.strftime("%Y-%m") != month or high.strftime("%Y-%m") != month)
                )
            ):
                raise ValueError(f"candidate keys or temporal range differs: {month}")
            coverage = coverage_month(database, features, statuses, candidate, a["rows"])
            totals[split]["a3_recent_normal_hours"] += coverage["rows"]
            for kind, discrete, numeric, admitted, count in coverage["groups"]:
                for counts in (totals[split], by_type[split][kind]):
                    counts["hours"] += count
                    counts[f"discrete_{discrete}"] += count
                    counts[f"numeric_{numeric}"] += count
                    counts["conditional_r3_hours"] += count if admitted else 0
            for field, values in coverage["reasons"].items():
                reasons[split][field].update(values)
            if sum(group[-1] for group in coverage["groups"] if group[3]) != key_count:
                raise ValueError("candidate coverage differs")
            directory = pending / f"year={month[:4]}" / f"month={month[5:]}"
            directory.mkdir(parents=True)
            qa_path = directory / "qa_counts.parquet"
            database.execute("CREATE OR REPLACE TEMP TABLE qa_month AS " + counts_sql())
            database.execute(
                "COPY (SELECT * FROM qa_month ORDER BY channel_id,prediction_time) TO "
                + quoted(str(qa_path))
                + " (FORMAT PARQUET,COMPRESSION ZSTD)"
            )
            patch = ",".join(
                f'CASE WHEN c.channel_id IS NOT NULL THEN c."{name}" ELSE f."{name}" '
                f'END AS "{name}"'
                for name in base_names
            )
            database.execute(
                "CREATE OR REPLACE TEMP TABLE model_month AS SELECT k.channel_id,"
                "k.prediction_time,"
                + patch
                + ","
                + ",".join(f'q."{name}"' for name in QA_NAMES)
                + " FROM keys k JOIN read_parquet(?) f USING(channel_id,prediction_time) "
                "LEFT JOIN corrections c USING(channel_id,prediction_time) "
                "JOIN qa_month q USING(channel_id,prediction_time)",
                [str(features)],
            )
            count, invalid_types = database.execute(
                "SELECT COUNT(*),COUNT(*) FILTER (WHERE m.sensor_type IS DISTINCT FROM k.sensor_type) "
                "FROM model_month m JOIN keys k USING(channel_id,prediction_time)"
            ).fetchone()
            if count != key_count or invalid_types:
                raise ValueError("model feature key count or sensor type differs")
            feature_path, gate_path = (
                directory / "model_features.parquet",
                directory / "qa_gate.parquet",
            )
            database.execute(
                "COPY (SELECT * FROM model_month ORDER BY channel_id,prediction_time) TO "
                + quoted(str(feature_path))
                + " (FORMAT PARQUET,COMPRESSION ZSTD)"
            )
            database.execute(
                "CREATE OR REPLACE TEMP TABLE qa_gate AS SELECT channel_id,prediction_time,"
                "CASE WHEN baseline_state_count=0 OR state_count_24h=0 OR "
                "state_transitions_24h IS NULL THEN 'unknown' ELSE 'eligible' END "
                "AS qa_discrete_data_status FROM model_month"
            )
            database.execute(
                "COPY (SELECT *,CASE WHEN qa_discrete_data_status='unknown' "
                "THEN 'qa_adjustment_removed_required_state_history' ELSE NULL END "
                "AS reason FROM qa_gate ORDER BY channel_id,prediction_time) TO "
                + quoted(str(gate_path))
                + " (FORMAT PARQUET,COMPRESSION ZSTD)"
            )
            changed_gate = database.execute(
                "SELECT COUNT(*) FROM qa_gate WHERE qa_discrete_data_status='unknown'"
            ).fetchone()[0]
            totals[split]["qa_gate_unknown"] += changed_gate
            support_values = database.execute(
                "SELECT "
                + ",".join(f'COUNT(*) FILTER(WHERE "{name}">0)' for name in QA_NAMES)
                + " FROM qa_month"
            ).fetchone()
            support[split].update(dict(zip(QA_NAMES, support_values, strict=True)))
            checks.extend(verify_python_samples(database, qa_path, m1_files))
            month_reports.append(
                {
                    "month": month,
                    "split": split,
                    "rows": key_count,
                    "a3_hours": coverage["rows"],
                    "qa_gate_unknown": changed_gate,
                    "files": {
                        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                        for path in (qa_path, feature_path, gate_path)
                    },
                }
            )
            print(
                json.dumps(
                    {"month": month, "model_rows": key_count, "qa_gate_unknown": changed_gate},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    report = {
        "schema_version": VERSION,
        "status": "a_train_validation_features_ready_b_review_pending",
        "source_contract_lf_sha256": frozen_rule_sha256(contract_path),
        "source_m1_manifest_sha256": contract["source_m1_manifest_sha256"],
        "source_a3_manifest_sha256": contract["source_a3_manifest_sha256"],
        "source_admission_manifest_sha256": contract["source_admission_manifest_sha256"],
        "source_qa_corrections_sha256": contract["source_qa_correction_features_sha256"],
        "qa_value_ruleset_version": QA_VALUE_RULESET_VERSION,
        "months": month_reports,
        "split_totals": {split: dict(counts) for split, counts in totals.items()},
        "past_data_reasons": {
            split: {field: dict(values) for field, values in fields.items()}
            for split, fields in reasons.items()
        },
        "by_type": {
            split: {kind: dict(counts) for kind, counts in kinds.items()}
            for split, kinds in by_type.items()
        },
        "qa_observed_category_rows_train_validation": qa_events,
        "qa_nonzero_feature_hours": {split: dict(counts) for split, counts in support.items()},
        "independent_python_checks": checks,
        "independent_qa_mismatches": 0,
        "source_files": source_files,
        "source_m1_files": [
            {"file": path.relative_to(m1_dir).as_posix(), "sha256": sha256(path)}
            for path in m1_files
        ],
        "code_lf_sha256": {
            name: frozen_rule_sha256(Path(__file__).resolve().parents[1] / name)
            for name in (
                "analysis/build_quality_improvement_a.py",
                "stage1/features/qa_values.py",
                "stage1/value_quality.py",
            )
        },
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": getattr(
                psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss
            ),
        },
        "target_read_for_features": False,
        "test_data_read": False,
        "new_metrics_computed": False,
        "training_ready": False,
        "population_expanded": False,
        "live_admission_approved": False,
        "limitations": [
            "R3 keys are a retrospective label-conditioned comparison cohort, not live admission.",
            "Coverage denominator is the A3 recent-Norma grid, not all possible channel hours.",
            "QA counts are diagnostic raw-observation counts, not confirmed physical failures.",
            "New features do not prove a metric improvement; B must validate their usefulness.",
            "Metric unit, QA gate and new independent test require joint review.",
        ],
    }
    (pending / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": VERSION,
        "status": report["status"],
        "source_contract_lf_sha256": report["source_contract_lf_sha256"],
        "report_sha256": sha256(pending / "report.json"),
        "allowlist_sha256": sha256(pending / "model_feature_allowlist.json"),
        "feature_count": 71,
        "chunk_count": len(month_reports),
        "row_count": sum(month["rows"] for month in month_reports),
        "chunks": month_reports,
    }
    (pending / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("m1-dir", "a3-dir", "admission-dir", "corrections-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--contract", type=Path, default=Path("ml/quality_improvement_feature_contract_v1.json")
    )
    parser.add_argument(
        "--base-allowlist", type=Path, default=Path("ml/r3_discrete_feature_allowlist_v1.json")
    )
    args = parser.parse_args()
    report = build(
        m1_dir=args.m1_dir,
        a3_dir=args.a3_dir,
        admission_dir=args.admission_dir,
        corrections_dir=args.corrections_dir,
        output_dir=args.output_dir,
        contract_path=args.contract,
        allowlist_path=args.base_allowlist,
    )
    print(json.dumps(report["split_totals"], ensure_ascii=False))


if __name__ == "__main__":
    main()
