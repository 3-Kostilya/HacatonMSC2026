"""Post-inference audit: compare shadow inputs with A3, never feed A3 to inference."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from analysis.replay_shadow_pilot import OUTPUT_SCHEMA
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.r6_rule import TERMS


def run(
    *, pilot_dir: Path, a3_dir: Path, freeze_path: Path, output_dir: Path, b_dir: Path | None = None
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest, freeze = read_json(pilot_dir / "manifest.json"), read_json(freeze_path)
    predictions = pilot_dir / manifest["prediction_file"]
    report_path = pilot_dir / "report.json"
    if (
        sha256(predictions) != manifest["prediction_sha256"]
        or sha256(report_path) != manifest["report_sha256"]
        or sha256(a3_dir / "manifest.json") != freeze["source_a3_manifest_sha256"]
        or not pq.read_schema(predictions).equals(OUTPUT_SCHEMA, check_metadata=False)
    ):
        raise ValueError("published pilot or A3 identity differs")
    pilot = read_json(report_path)
    a3 = read_json(a3_dir / "manifest.json")
    if (
        pilot["source_freeze_lf_sha256"] != frozen_rule_sha256(freeze_path)
        or pilot["source_m1_manifest_sha256"] != a3["source_m1_manifest_sha256"]
    ):
        raise ValueError("pilot lineage differs from frozen rule or A3")
    start, end = (
        datetime.fromisoformat(pilot["start"]),
        datetime.fromisoformat(pilot["end_exclusive"]),
    )
    formula = "+".join(f'{weight}*p."{name}"' for name, weight in TERMS.items())
    months = []
    b_check = None
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        database.execute("CREATE TEMP TABLE p AS SELECT * FROM read_parquet(?)", [str(predictions)])
        total, unique, invalid, scored = database.execute(
            f"""SELECT COUNT(*), COUNT(DISTINCT(channel_id,prediction_time)),
                COUNT(*) FILTER(WHERE availability_status IS DISTINCT FROM 'unknown'
                    OR channel_id IS NULL OR prediction_time IS NULL
                    OR admission_status IS NULL OR admission_status NOT IN ('eligible','unknown','excluded')
                    OR admission_reasons IS NULL OR warning_emitted IS NULL
                    OR (admission_status='eligible' AND len(admission_reasons)<>0)
                    OR (admission_status<>'eligible' AND len(admission_reasons)=0)
                    OR (admission_status='eligible' AND
                        (rule_score IS NULL OR NOT isfinite(rule_score) OR rule_score IS DISTINCT FROM ({formula})
                         OR above_frozen_threshold IS DISTINCT FROM (rule_score>=7.1)))
                    OR (admission_status<>'eligible' AND
                        (rule_score IS NOT NULL OR above_frozen_threshold IS NOT NULL OR warning_emitted))
                    OR (warning_emitted AND above_frozen_threshold IS DISTINCT FROM TRUE))
                , COUNT(*) FILTER(WHERE admission_status='eligible') FROM p"""
        ).fetchone()
        if (
            total != unique
            or total != manifest["prediction_rows"]
            or invalid
            or scored != pilot["conditionally_scored_hours"]
        ):
            raise ValueError("pilot keys, score or missing-prediction contract differs")
        if b_dir is not None:
            b_manifest = read_json(b_dir / "manifest.json")
            b_report = read_json(b_dir / "report.json")
            b_predictions = b_dir / "shadow_decisions.parquet"
            if (
                sha256(b_predictions) != b_manifest["decisions_sha256"]
                or sha256(b_dir / "report.json") != b_manifest["report_sha256"]
                or sha256(b_dir / "checkpoint.json") != b_manifest["checkpoint_sha256"]
                or b_report["source_input_sha256"] != manifest["b_input_sha256"]
                or sha256(pilot_dir / manifest["b_input_file"]) != manifest["b_input_sha256"]
                or b_report["source_freeze_sha256"] != pilot["source_freeze_lf_sha256"]
            ):
                raise ValueError("B shadow output lineage differs")
            b_rows, b_unique, b_mismatches = database.execute(
                """SELECT COUNT(*), COUNT(DISTINCT (b.channel_id,b.prediction_time)),
                    COUNT(*) FILTER(WHERE p.channel_id IS NULL OR b.channel_id IS NULL
                      OR p.admission_status IS DISTINCT FROM b.admission_status
                      OR COALESCE(p.sensor_type,'<unknown_or_conflicting>') IS DISTINCT FROM b.sensor_type
                      OR p.rule_score IS DISTINCT FROM b.rule_score
                      OR p.above_frozen_threshold IS DISTINCT FROM b.threshold_crossed
                      OR p.warning_emitted IS DISTINCT FROM b.shadow_warning
                      OR b.automatic_action_taken IS DISTINCT FROM FALSE
                      OR b.delivery_mode IS DISTINCT FROM 'record_only')
                    FROM p FULL OUTER JOIN read_parquet(?) b
                      ON p.channel_id=b.channel_id
                     AND p.prediction_time=CAST(b.prediction_time AS TIMESTAMP)""",
                [str(b_predictions)],
            ).fetchone()
            if b_rows != total or b_unique != total or b_mismatches:
                raise ValueError("A/B shadow decisions differ")
            b_check = {
                "source_b_manifest_sha256": sha256(b_dir / "manifest.json"),
                "rows_checked": b_rows,
                "decision_mismatches": b_mismatches,
                "delivery_mode": "record_only",
            }
        for item in a3["chunks"]:
            month_start = datetime.fromisoformat(item["start_at"])
            month_end = datetime.fromisoformat(item["end_at"])
            if month_start >= end or month_end <= start:
                continue
            features, statuses = a3_dir / item["features_file"], a3_dir / item["row_status_file"]
            if (
                sha256(features) != item["features_sha256"]
                or sha256(statuses) != item["row_status_sha256"]
            ):
                raise ValueError("A3 month files differ")
            conditions = ",".join(
                f'COUNT(*) FILTER(WHERE f.channel_id IS NOT NULL AND p."{name}" IS DISTINCT FROM f."{name}")'
                for name in TERMS
            )
            result = database.execute(
                """SELECT COUNT(*), COUNT(*) FILTER(WHERE f.channel_id IS NOT NULL),
                    COUNT(*) FILTER(WHERE p.admission_status='eligible' AND
                        (f.channel_id IS NULL OR s.discrete_data_status IS DISTINCT FROM 'eligible')), """
                + conditions
                + """ FROM p LEFT JOIN read_parquet(?) f USING(channel_id,prediction_time)
                    LEFT JOIN read_parquet(?) s USING(channel_id,prediction_time)
                    WHERE p.prediction_time>=? AND p.prediction_time<?""",
                [str(features), str(statuses), month_start, month_end],
            ).fetchone()
            differences = dict(zip(TERMS, result[3:], strict=True))
            if result[2] or any(differences.values()):
                raise ValueError(f"shadow/A3 past inputs differ: {item['month']}: {result}")
            months.append(
                {
                    "month": item["month"],
                    "pilot_rows": result[0],
                    "common_a3_rows": result[1],
                    "eligible_missing_or_bad_legacy_gate": result[2],
                    "input_mismatches": differences,
                }
            )
    if sum(item["pilot_rows"] for item in months) != total:
        raise ValueError("A3 audit month coverage differs")
    audit = {
        "schema_version": "shadow-a-post-inference-a3-audit-v1",
        "status": "passed",
        "source_pilot_manifest_sha256": sha256(pilot_dir / "manifest.json"),
        "source_a3_manifest_sha256": sha256(a3_dir / "manifest.json"),
        "rows_checked": total,
        "common_a3_rows": sum(item["common_a3_rows"] for item in months),
        "conditionally_scored_hours": pilot["conditionally_scored_hours"],
        "months": months,
        "scope": "Separate post-inference audit of raw rule inputs and legacy past data gate; no target or quality metrics.",
        "joint_pilot_acceptance": False,
        "b_decision_check": b_check,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, default=Path("ml/r6_frozen_rule_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--b-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                pilot_dir=args.pilot_dir,
                a3_dir=args.a3_dir,
                freeze_path=args.freeze,
                output_dir=args.output_dir,
                b_dir=args.b_dir,
            )
        )
    )


if __name__ == "__main__":
    main()
