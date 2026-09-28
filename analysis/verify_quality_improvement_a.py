"""Check exported Q1 features, QA gates and sampled base/correction overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_quality_improvement_a import QA_NAMES, expected_months, safe_path
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import read_json, sha256
from stage1.features.qa_values import QA_CATEGORIES
from stage1.features.schema import WINDOW_HOURS


def validate_month(database, model: Path, qa: Path, gate: Path, names: list[str]) -> int:
    schemas = (
        (model, ["channel_id", "prediction_time", *names]),
        (qa, ["channel_id", "prediction_time", *QA_NAMES]),
        (gate, ["channel_id", "prediction_time", "qa_discrete_data_status", "reason"]),
    )
    counts = []
    for table, (path, columns) in zip(("model", "qa", "gate"), schemas, strict=True):
        schema = pq.read_schema(path)
        if schema.names != columns:
            raise ValueError(f"exported schema differs: {path}")
        if table in ("model", "qa") and any(
            schema.field(name).type != pa.int64() for name in QA_NAMES
        ):
            raise ValueError("QA counts must have integer type")
        literal = "'" + str(path).replace("'", "''") + "'"
        database.execute(
            f"CREATE OR REPLACE TEMP VIEW {table} AS SELECT * FROM read_parquet({literal})"
        )
        count, unique, missing = database.execute(
            f"SELECT COUNT(*),COUNT(DISTINCT (channel_id,prediction_time)),"
            f"COUNT(*) FILTER(WHERE channel_id IS NULL OR prediction_time IS NULL) FROM {table}"
        ).fetchone()
        if count != unique or missing:
            raise ValueError(f"duplicate or missing exported key: {table}")
        counts.append(count)
    if len(set(counts)) != 1:
        raise ValueError("exported table row counts differ")
    invalid = [f'qa."{name}" IS NULL OR qa."{name}"<0' for name in QA_NAMES]
    for category in QA_CATEGORIES:
        for shorter, longer in zip(WINDOW_HOURS, WINDOW_HOURS[1:]):
            invalid.append(
                f'qa."qa_{category}_count_{shorter}h">qa."qa_{category}_count_{longer}h"'
            )
    if database.execute("SELECT COUNT(*) FROM qa WHERE " + " OR ".join(invalid)).fetchone()[0]:
        raise ValueError("QA counts violate nonnegative nested windows")
    comparisons = [f'm."{name}" IS DISTINCT FROM q."{name}"' for name in QA_NAMES]
    expected_status = (
        "CASE WHEN m.baseline_state_count=0 OR m.state_count_24h=0 OR "
        "m.state_transitions_24h IS NULL THEN 'unknown' ELSE 'eligible' END"
    )
    comparisons += [
        "m.channel_id IS NULL OR q.channel_id IS NULL OR g.channel_id IS NULL",
        f"g.qa_discrete_data_status IS DISTINCT FROM ({expected_status})",
        "g.reason IS DISTINCT FROM (CASE WHEN "
        + expected_status
        + "='unknown' THEN 'qa_adjustment_removed_required_state_history' ELSE NULL END)",
    ]
    bad = database.execute(
        "SELECT COUNT(*) FROM model m FULL JOIN qa q USING(channel_id,prediction_time) "
        "FULL JOIN gate g USING(channel_id,prediction_time) WHERE " + " OR ".join(comparisons)
    ).fetchone()[0]
    if bad:
        raise ValueError("exported model/QA/gate keys or values differ")
    return counts[0]


def verify(*, package: Path, a3_dir: Path, corrections_dir: Path, output: Path) -> dict:
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    manifest = read_json(package / "manifest.json")
    if sha256(package / "report.json") != manifest["report_sha256"]:
        raise ValueError("package report hash differs")
    if sha256(package / "model_feature_allowlist.json") != manifest["allowlist_sha256"]:
        raise ValueError("package allowlist hash differs")
    report = read_json(package / "report.json")
    allowlist = read_json(package / "model_feature_allowlist.json")
    if sha256(a3_dir / "manifest.json") != report["source_a3_manifest_sha256"]:
        raise ValueError("A3 provenance differs")
    corrections = corrections_dir / "feature_corrections.parquet"
    if sha256(corrections) != report["source_qa_corrections_sha256"]:
        raise ValueError("QA overlay provenance differs")
    if [chunk["month"] for chunk in manifest["chunks"]] != expected_months():
        raise ValueError("package months differ or include test")
    if manifest["feature_count"] != 71 or len(allowlist["feature_names"]) != 71:
        raise ValueError("feature count differs")
    source = {chunk["month"]: chunk for chunk in read_json(a3_dir / "manifest.json")["chunks"]}
    results = []
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        database.execute("SET memory_limit='1GB'")
        database.execute(
            "CREATE TEMP TABLE corrections AS SELECT * FROM read_parquet(?) "
            "WHERE prediction_time<TIMESTAMP '2026-01-01' AND year(prediction_time)<>2021",
            [str(corrections)],
        )
        for chunk in manifest["chunks"]:
            month = chunk["month"]
            directory = package / f"year={month[:4]}" / f"month={month[5:]}"
            paths = [
                directory / name
                for name in ("model_features.parquet", "qa_counts.parquet", "qa_gate.parquet")
            ]
            for path in paths:
                metadata = chunk["files"][path.name]
                if sha256(path) != metadata["sha256"] or path.stat().st_size != metadata["bytes"]:
                    raise ValueError(f"exported file hash or size differs: {path}")
            count = validate_month(database, *paths, allowlist["feature_names"])
            if count != chunk["rows"]:
                raise ValueError("exported monthly count differs")
            # Compare all corrected keys and two stable uncorrected keys against original A3.
            database.execute(
                "CREATE OR REPLACE TEMP TABLE sample AS SELECT channel_id,prediction_time FROM "
                "(SELECT m.channel_id,m.prediction_time FROM model m JOIN corrections c "
                "USING(channel_id,prediction_time) UNION SELECT channel_id,prediction_time FROM "
                "(SELECT m.channel_id,m.prediction_time FROM model m LEFT JOIN corrections c "
                "USING(channel_id,prediction_time) WHERE c.channel_id IS NULL ORDER BY "
                "sha256(m.channel_id || CAST(m.prediction_time AS VARCHAR)) LIMIT 2))"
            )
            base_file = safe_path(a3_dir, source[month]["features_file"], month=month)
            expressions = [
                f'm."{name}" IS DISTINCT FROM (CASE WHEN c.channel_id IS NOT NULL '
                f'THEN c."{name}" ELSE f."{name}" END)'
                for name in allowlist["base_feature_names"]
            ]
            original_differences = [
                f'm."{name}" IS DISTINCT FROM f."{name}"'
                for name in allowlist["base_feature_names"]
            ]
            rows, mismatches, overlay_rows, changed_base_rows = database.execute(
                "SELECT COUNT(*),COUNT(*) FILTER (WHERE "
                + " OR ".join(expressions)
                + "),COUNT(*) FILTER(WHERE c.channel_id IS NOT NULL),"
                "COUNT(*) FILTER(WHERE "
                + " OR ".join(original_differences)
                + ") FROM sample s JOIN model m USING(channel_id,prediction_time) "
                "LEFT JOIN read_parquet(?) f USING(channel_id,prediction_time) "
                "LEFT JOIN corrections c USING(channel_id,prediction_time)",
                [str(base_file)],
            ).fetchone()
            if rows != database.execute("SELECT COUNT(*) FROM sample").fetchone()[0] or mismatches:
                raise ValueError("exported QA overlay differs from A3/corrections")
            nonzero_hours = database.execute(
                "SELECT COUNT(*) FROM model WHERE "
                + " OR ".join(f'"{name}">0' for name in QA_NAMES)
            ).fetchone()[0]
            results.append(
                {
                    "month": month,
                    "rows": count,
                    "base_overlay_sample_rows": rows,
                    "base_overlay_mismatches": mismatches,
                    "overlay_keys_present": overlay_rows,
                    "base_feature_changed_sample_rows": changed_base_rows,
                    "qa_any_nonzero_hours": nonzero_hours,
                }
            )
            print(json.dumps({"verified_month": month, "rows": count}), flush=True)
    total = sum(row["rows"] for row in results)
    if total != manifest["row_count"] or total != sum(
        part["conditional_r3_hours"] for part in report["split_totals"].values()
    ):
        raise ValueError("package total differs")
    result = {
        "status": "a_export_consistency_verified_b_independent_review_pending",
        "source_manifest_sha256": sha256(package / "manifest.json"),
        "validator_lf_sha256": frozen_rule_sha256(Path(__file__)),
        "row_count": total,
        "months": results,
        "base_overlay_sample_rows": sum(row["base_overlay_sample_rows"] for row in results),
        "overlay_keys_present": sum(row["overlay_keys_present"] for row in results),
        "base_feature_changed_sample_rows": sum(
            row["base_feature_changed_sample_rows"] for row in results
        ),
        "qa_any_nonzero_hours_by_split": {
            split: sum(
                row["qa_any_nonzero_hours"]
                for row in results
                if (row["month"].startswith("2025")) == (split == "validation")
            )
            for split in ("train", "validation")
        },
        "mismatches": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "test_rows_used": False,
        "new_model_quality_verified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "a3-dir", "corrections-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        package=args.package,
        a3_dir=args.a3_dir,
        corrections_dir=args.corrections_dir,
        output=args.output,
    )
    print(json.dumps({"verified_rows": result["row_count"], "mismatches": result["mismatches"]}))


if __name__ == "__main__":
    main()
