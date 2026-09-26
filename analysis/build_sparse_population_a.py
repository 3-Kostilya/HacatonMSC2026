"""All 72-month causal candidate decisions and missing-aware features; labels audited later."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import time

import duckdb
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_a2_hourly import _monthly_files
from analysis.build_quality_improvement_a import YEARS, QA_NAMES, expected_months
from analysis.build_quality_improvement_a import create_qa_prefixes, counts_sql, quoted, safe_path
from analysis.r6_provenance import frozen_rule_sha256
from analysis.sparse_population_a import build_past_prefixes, decision_sql, finalize_sql
from analysis.sparse_population_a import retain_full_prefixes, slice_month_prefixes
from analysis.train_r4_discrete_baselines import read_json, sha256
from stage1.features.sparse_admission import VERSION as ADMISSION_VERSION


VERSION = "q2-a-full-sparse-population-v1"
CODE = (
    "analysis/build_sparse_population_a.py",
    "analysis/sparse_population_a.py",
    "analysis/build_quality_improvement_a.py",
    "stage1/features/sparse_admission.py",
    "stage1/shadow/stream.py",
    "stage1/state_labeling/registered_episodes.py",
    "stage1/state_labeling/operational.py",
    "stage1/state_labeling/rules.py",
    "stage1/value_quality.py",
)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_month(db, expected, output_file, feature_file, feature_names):
    count, unique, violations = db.execute(
        "SELECT COUNT(*),COUNT(DISTINCT (channel_id,prediction_time)),COUNT(*) FILTER(WHERE "
        "admission_evidence_through>prediction_time OR last_explicit_normal_at>prediction_time "
        "OR first_usable_at>prediction_time OR second_usable_at>prediction_time "
        "OR (admission_status='eligible' AND (len(admission_reasons)>0 OR blocking_qa_count_24h>0)) "
        "OR availability_status<>'unknown') FROM read_parquet(?)",
        [str(output_file)],
    ).fetchone()
    if count != expected or unique != count or violations:
        raise ValueError("full candidate grid count, duplicate or causal guard violation")
    feature_count, feature_unique, unexpected = db.execute(
        "SELECT COUNT(*),COUNT(DISTINCT (f.channel_id,f.prediction_time)),"
        "COUNT(*) FILTER(WHERE a.channel_id IS NULL OR a.admission_status<>'eligible' "
        "OR f.sensor_type IS DISTINCT FROM a.sensor_type) FROM read_parquet(?) f "
        "LEFT JOIN read_parquet(?) a USING(channel_id,prediction_time)",
        [str(feature_file), str(output_file)],
    ).fetchone()
    eligible = db.execute(
        "SELECT COUNT(*) FROM decisions WHERE admission_status='eligible'"
    ).fetchone()[0]
    if feature_count != eligible or feature_unique != eligible or unexpected:
        raise ValueError("full missing-aware features do not exactly cover eligible hours")
    if pq.ParquetFile(feature_file).schema_arrow.names != [
        "channel_id",
        "prediction_time",
        *feature_names,
    ]:
        raise ValueError("full feature-only schema differs from allowlist")
    return {
        "decision_rows": count,
        "feature_rows": eligible,
        "key_mismatches": 0,
        "causal_violations": 0,
    }


def build(*, m1_dir, a3_dir, b3_dir, corrections_dir, output_dir):
    begun = time.perf_counter()
    q1 = read_json(Path("ml/quality_improvement_feature_contract_v1.json"))
    base = read_json(Path("ml/r3_discrete_feature_allowlist_v1.json"))
    pins = {
        "m1": q1["source_m1_manifest_sha256"],
        "a3": q1["source_a3_manifest_sha256"],
        "b3": base["source_b3_manifest_sha256"],
        "corrections": q1["source_qa_correction_features_sha256"],
    }
    for path, pin in (
        (m1_dir / "manifest.json", pins["m1"]),
        (a3_dir / "manifest.json", pins["a3"]),
        (b3_dir / "manifest.json", pins["b3"]),
        (corrections_dir / "feature_corrections.parquet", pins["corrections"]),
    ):
        if sha256(path) != pin:
            raise ValueError("full candidate source manifest or corrections differ")
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or pending.exists():
        raise FileExistsError(output_dir)
    a3 = read_json(a3_dir / "manifest.json")
    chunks = {c["month"]: c for c in a3["chunks"]}
    m1_files = []
    for year in YEARS:
        files, missing = _monthly_files(m1_dir, datetime(year, 1, 1), datetime(year + 1, 1, 1))
        if missing:
            raise ValueError(f"source months absent: {missing}")
        m1_files.extend(files)
    if any("year=2026" in p.parts or "year=2021" in p.parts for p in m1_files):
        raise ValueError("test/excluded-year file in candidate calculation")
    base_names = base["feature_names"]
    masks = [f"missing__{name}" for name in base_names if name != "sensor_type"]
    feature_names = [*base_names, *QA_NAMES, *masks]
    pending.mkdir(parents=True)
    allowlist = {
        "schema_version": VERSION,
        "admission_version": ADMISSION_VERSION,
        "feature_names": feature_names,
        "base_feature_names": base_names,
        "qa_feature_names": list(QA_NAMES),
        "missingness_feature_names": masks,
        "feature_count": len(feature_names),
        "categorical_feature_names": ["sensor_type"],
        "keys_not_features": ["channel_id", "prediction_time"],
        "diagnostics_and_labels_not_features": True,
        "training_ready": False,
        "independent_b_review_required": True,
        "fit_scope": "train_only",
    }
    write_json(pending / "model_feature_allowlist.json", allowlist)
    proofs = []
    months = []
    totals = defaultdict(Counter)
    with duckdb.connect(config={"temp_directory": str(pending / "db-spill")}) as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='3GB'")
        db.execute("SET preserve_insertion_order=false")
        print("building full causal M1 state prefixes", flush=True)
        source_summary = build_past_prefixes(db, [str(p) for p in m1_files])
        print(
            json.dumps(
                {"past_prefixes_complete": source_summary, "seconds": time.perf_counter() - begun}
            ),
            flush=True,
        )
        qa_summary = create_qa_prefixes(db, m1_files)
        retain_full_prefixes(db)
        db.execute(
            "CREATE TEMP TABLE corrections AS SELECT * FROM read_parquet(?) "
            "WHERE prediction_time<TIMESTAMP '2026-01-01' AND year(prediction_time)<>2021",
            [str(corrections_dir / "feature_corrections.parquet")],
        )
        if db.execute(
            "SELECT COUNT(*) FROM (SELECT channel_id,prediction_time FROM corrections "
            "GROUP BY ALL HAVING COUNT(*)>1)"
        ).fetchone()[0]:
            raise ValueError("duplicate QA correction key")
        for month in expected_months():
            started = time.perf_counter()
            slice_month_prefixes(db, month)
            source = chunks[month]
            month_manifest = safe_path(a3_dir, source["manifest_file"], month=month)
            features = safe_path(a3_dir, source["features_file"], month=month)
            for path, pin in (
                (month_manifest, source["manifest_sha256"]),
                (features, source["features_sha256"]),
            ):
                if sha256(path) != pin:
                    raise ValueError(f"A3 source hash differs for {month}")
                proofs.append({"month": month, "file": path.name, "sha256": pin})
            db.execute(
                "CREATE OR REPLACE TEMP TABLE keys AS SELECT channel_id,prediction_time,"
                "CASE WHEN year(prediction_time)<2021 THEN 0 ELSE 1 END AS archive_segment "
                "FROM read_parquet(?)",
                [str(features)],
            )
            if db.execute(
                "SELECT COUNT(*) FROM keys WHERE strftime(prediction_time,'%Y-%m')<>?", [month]
            ).fetchone()[0]:
                raise ValueError("A3 grid point outside declared month")
            db.execute("CREATE OR REPLACE TEMP TABLE decisions_without_qa AS " + decision_sql())
            db.execute("CREATE OR REPLACE TEMP TABLE qa_month AS " + counts_sql())
            db.execute("CREATE OR REPLACE TEMP TABLE decisions AS " + finalize_sql())
            directory = pending / f"year={month[:4]}" / f"month={month[5:]}"
            directory.mkdir(parents=True)
            output_file = directory / "admission.parquet"
            db.execute(
                "COPY (SELECT * EXCLUDE(" + ",".join(QA_NAMES) + ") FROM decisions "
                "ORDER BY channel_id,prediction_time) TO "
                + quoted(str(output_file))
                + " (FORMAT PARQUET,COMPRESSION ZSTD)"
            )
            patch = ",".join(
                f'CASE WHEN c.channel_id IS NOT NULL THEN c."{name}" ELSE f."{name}" '
                f'END AS "{name}"'
                for name in base_names
            )
            db.execute(
                "CREATE OR REPLACE TEMP TABLE base_features AS SELECT d.channel_id,"
                "d.prediction_time,"
                + patch
                + ","
                + ",".join(f'd."{n}"' for n in QA_NAMES)
                + " FROM decisions d JOIN read_parquet(?) f USING(channel_id,prediction_time) "
                "LEFT JOIN corrections c USING(channel_id,prediction_time) "
                "WHERE d.admission_status='eligible'",
                [str(features)],
            )
            feature_file = directory / "model_features.parquet"
            missing_sql = ",".join(
                f'CAST("{n}" IS NULL AS TINYINT) AS "missing__{n}"'
                for n in base_names
                if n != "sensor_type"
            )
            db.execute(
                "COPY (SELECT *," + missing_sql + " FROM base_features "
                "ORDER BY channel_id,prediction_time) TO "
                + quoted(str(feature_file))
                + " (FORMAT PARQUET,COMPRESSION ZSTD)"
            )
            verification = verify_month(
                db, source["rows"], output_file, feature_file, feature_names
            )
            groups = db.execute(
                "SELECT COALESCE(sensor_type,'<unknown_or_conflicting>'),"
                "admission_status_without_qa,admission_status,COUNT(*) "
                "FROM decisions GROUP BY ALL"
            ).fetchall()
            reasons = dict(
                db.execute(
                    "SELECT reason,COUNT(*) FROM decisions,"
                    "UNNEST(admission_reasons) u(reason) GROUP BY reason"
                ).fetchall()
            )
            split = "validation" if month[:4] == "2025" else "train"
            for _, before, after, count in groups:
                totals[split][f"candidate_{after}"] += count
                totals[split][f"before_qa_{before}"] += count
            correction_rows = db.execute(
                "SELECT COUNT(*) FROM base_features b JOIN corrections c "
                "USING(channel_id,prediction_time)"
            ).fetchone()[0]
            report = {
                "schema_version": VERSION,
                "month": month,
                "split": split,
                **verification,
                "groups": [
                    {"sensor_type": k, "without_qa": p, "candidate": a, "hours": n}
                    for k, p, a, n in groups
                ],
                "reason_counts": reasons,
                "qa_correction_keys_in_features": correction_rows,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "files": {
                    p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                    for p in (output_file, feature_file)
                },
            }
            write_json(directory / "manifest.json", report)
            months.append(
                {
                    **report,
                    "manifest_file": (directory / "manifest.json").relative_to(pending).as_posix(),
                    "manifest_sha256": sha256(directory / "manifest.json"),
                }
            )
            print(
                json.dumps({"month": month, **verification, "seconds": report["elapsed_seconds"]}),
                flush=True,
            )
    # The causal phase is closed before any label contents are read.
    diagnosis = audit_labels(pending, b3_dir, a3_dir)
    source_files = [
        {"file": p.relative_to(m1_dir).as_posix(), "sha256": sha256(p)} for p in m1_files
    ]
    memory = psutil.Process().memory_info()
    report = {
        "schema_version": VERSION,
        "status": "full_candidate_features_complete_b_review_pending",
        "admission_version": ADMISSION_VERSION,
        "source_manifests": pins,
        "months": months,
        "source_m1_files": source_files,
        "source_a3_files": proofs,
        "code_lf_sha256": {name: frozen_rule_sha256(Path(name)) for name in CODE},
        "source_proposal_lf_sha256": frozen_rule_sha256(
            Path("ml/sparse_admission_proposal_v1.json")
        ),
        "source_b_review_lf_sha256": frozen_rule_sha256(
            Path("docs/ml-q2-b-sparse-admission-review.md")
        ),
        "source_summary": source_summary,
        "qa_category_events": qa_summary,
        "split_totals": {s: dict(v) for s, v in totals.items()},
        "label_audit": diagnosis,
        "feature_count": len(feature_names),
        "all_expected_months": expected_months(),
        "labels_read_for_inference": False,
        "labels_unchanged": True,
        "test_events_read": False,
        "frozen_r6_changed": False,
        "training_ready": False,
        "joint_full_population_approved": False,
        "new_model_metrics_computed": False,
        "physical_failure_claim": False,
        "resources": {
            "elapsed_seconds": round(time.perf_counter() - begun, 3),
            "peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss),
        },
    }
    write_json(pending / "report.json", report)
    manifest = {
        "schema_version": VERSION,
        "status": report["status"],
        "training_ready": False,
        "source_manifests": pins,
        "months": months,
        "files": {
            name: {"sha256": sha256(pending / name), "bytes": (pending / name).stat().st_size}
            for name in (
                "report.json",
                "model_feature_allowlist.json",
                "positive_hour_diagnostics.parquet",
                "episode_diagnostics.parquet",
            )
        },
    }
    write_json(pending / "manifest.json", manifest)
    pending.rename(output_dir)
    return report


def audit_labels(package, b3_dir, a3_dir):
    """Separate retrospective diagnostics. These rows never feed feature inference."""
    b3 = read_json(b3_dir / "manifest.json")
    a3 = read_json(a3_dir / "manifest.json")
    b_chunks = {c["month"]: c for c in b3["chunks"]}
    a_chunks = {c["month"]: c for c in a3["chunks"]}
    positive = []
    for month in expected_months():
        b = b_chunks[month]
        bm = safe_path(b3_dir, b["manifest_file"], month=month)
        if sha256(bm) != b["manifest_sha256"]:
            raise ValueError("B3 month manifest differs")
        meta = read_json(bm)
        labels = bm.parent / "registered_forecast_labels.parquet"
        if (
            meta["source_a3_month_manifest_sha256"] != a_chunks[month]["manifest_sha256"]
            or sha256(labels) != meta["files"][labels.name]["sha256"]
        ):
            raise ValueError("B3 label lineage or hash differs")
        status = package / f"year={month[:4]}" / f"month={month[5:]}" / "admission.parquet"
        with duckdb.connect() as db:
            rows = (
                db.execute(
                    "SELECT l.*,d.* EXCLUDE(channel_id,prediction_time,sensor_type),"
                    "? AS prediction_month FROM read_parquet(?) l "
                    "JOIN read_parquet(?) d USING(channel_id,prediction_time) "
                    "WHERE l.target=1 AND l.label_status='positive' "
                    "AND l.split_status='assigned' AND l.split IN ('train','validation')",
                    [month, str(labels), str(status)],
                )
                .to_arrow_table()
                .to_pylist()
            )
            positive.extend(rows)
    episodes = {}
    for row in positive:
        key = row["split"], row["target_episode_id"]
        item = episodes.setdefault(
            key,
            {
                "split": key[0],
                "target_episode_id": key[1],
                "sensor_type": row["sensor_type"],
                "channel_id": row["channel_id"],
                "positive_hours": 0,
                "candidate_hours": 0,
                "without_qa_hours": 0,
                "reason_counts": Counter(),
                "months": set(),
            },
        )
        if (item["sensor_type"], item["channel_id"]) != (row["sensor_type"], row["channel_id"]):
            raise ValueError("episode channel/type mismatch")
        item["positive_hours"] += 1
        item["candidate_hours"] += int(row["admission_status"] == "eligible")
        item["without_qa_hours"] += int(row["admission_status_without_qa"] == "eligible")
        item["reason_counts"].update(set(row["admission_reasons"]))
        item["months"].add(row["prediction_month"])
    full = read_json(b3_dir / "report.json")
    if sha256(b3_dir / "report.json") != b3["report_sha256"]:
        raise ValueError("full B3 report differs")
    by_split = defaultdict(Counter)
    by_type = defaultdict(lambda: defaultdict(Counter))
    by_month = defaultdict(Counter)
    loss_reasons = defaultdict(Counter)
    output = []
    for item in episodes.values():
        available = item["candidate_hours"] > 0
        before = item["without_qa_hours"] > 0
        metrics = {
            "all_episodes": 1,
            "candidate_available_episodes": int(available),
            "without_qa_available_episodes": int(before),
            "lost_solely_to_qa_episodes": int(before and not available),
        }
        by_split[item["split"]].update(metrics)
        by_type[item["split"]][item["sensor_type"] or "<unknown>"].update(metrics)
        for month in item["months"]:
            # This month metric is refined below from that month's own hours.
            by_month[month]["all_episodes_with_assigned_hour"] += 1
        every = sorted(
            reason
            for reason, count in item["reason_counts"].items()
            if count == item["positive_hours"]
        )
        if not available:
            loss_reasons[item["split"]].update(every)
        output.append(
            {
                **{k: v for k, v in item.items() if k not in {"reason_counts", "months"}},
                "prediction_months": sorted(item["months"]),
                "reasons_every_positive_hour": every,
                "reasons_any_positive_hour": sorted(item["reason_counts"]),
            }
        )
    for split, total in by_split.items():
        if total["all_episodes"] != full["assigned_unique_positive_episodes_by_split"][split]:
            raise ValueError("full assigned episode denominator differs")
    for month in expected_months():
        rows = [r for r in positive if r["prediction_month"] == month]
        by_month[month]["candidate_available_episodes"] = len(
            {r["target_episode_id"] for r in rows if r["admission_status"] == "eligible"}
        )
        by_month[month]["without_qa_available_episodes"] = len(
            {r["target_episode_id"] for r in rows if r["admission_status_without_qa"] == "eligible"}
        )
    pq.write_table(
        pa.Table.from_pylist(positive),
        package / "positive_hour_diagnostics.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(output), package / "episode_diagnostics.parquet", compression="zstd"
    )
    return {
        "positive_hours": len(positive),
        "episodes": len(episodes),
        "by_split": {s: dict(v) for s, v in by_split.items()},
        "by_type": {s: {k: dict(v) for k, v in kinds.items()} for s, kinds in by_type.items()},
        "by_prediction_month": {m: dict(v) for m, v in by_month.items()},
        "missed_episode_reasons_every_positive_hour": {s: dict(v) for s, v in loss_reasons.items()},
        "retrospective_diagnostics_not_model_inputs": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("m1-dir", "a3-dir", "b3-dir", "corrections-dir", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    result = build(
        m1_dir=args.m1_dir,
        a3_dir=args.a3_dir,
        b3_dir=args.b3_dir,
        corrections_dir=args.corrections_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "split_totals": result["split_totals"],
                "episodes": result["label_audit"]["by_split"],
                "resources": result["resources"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
