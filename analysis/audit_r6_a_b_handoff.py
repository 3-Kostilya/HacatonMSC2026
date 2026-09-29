"""Verify transferred B originals against A's pinned raw-history reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import read_json, sha256


MONTHS = tuple(f"2026-{month:02d}" for month in range(1, 7))
KEYS = ("channel_id", "prediction_time")


def verified_file(directory: Path, name: str, expected_hash: str) -> Path:
    """Do not trust handoff paths or hashes without checking the actual files."""
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"file escapes package: {name}")
    if sha256(path) != expected_hash:
        raise ValueError(f"file hash differs: {name}")
    return path


def verified_package(directory: Path, expected_hash: str) -> tuple[dict, dict]:
    manifest = read_json(verified_file(directory, "manifest.json", expected_hash))
    report = read_json(verified_file(directory, "report.json", manifest["report_sha256"]))
    entries = manifest["monthly_predictions"]
    if sorted(item["month"] for item in entries) != list(MONTHS):
        raise ValueError("monthly prediction coverage differs from frozen test")
    for item in entries:
        verified_file(directory, item["file"], item["sha256"])
        if "history_inputs_file" in item:
            verified_file(directory, item["history_inputs_file"], item["history_inputs_sha256"])
    verified_file(directory, "emitted_alerts.parquet", manifest["emitted_alerts_sha256"])
    return manifest, report


def compare_parquets(database: duckdb.DuckDBPyConnection, left: Path, right: Path) -> dict:
    """Exact, null-safe logical comparison, independent of row order and metadata."""
    schema = pq.read_schema(left)
    if not schema.equals(pq.read_schema(right), check_metadata=False):
        raise ValueError("Parquet logical schemas differ")
    if not all(key in schema.names for key in KEYS):
        raise ValueError("prediction keys absent")
    counts = []
    for path in (left, right):
        rows, distinct_keys, null_keys = database.execute(
            """SELECT COUNT(*), COUNT(DISTINCT (channel_id,prediction_time)),
                      COUNT(*) FILTER (WHERE channel_id IS NULL OR prediction_time IS NULL)
               FROM read_parquet(?)""",
            [str(path)],
        ).fetchone()
        if rows != distinct_keys or null_keys:
            raise ValueError("duplicate or null prediction keys")
        counts.append(rows)
    value_columns = [name for name in schema.names if name not in KEYS]
    conditions = []
    for name in value_columns:
        quoted = '"' + name.replace('"', '""') + '"'
        conditions.append(f"COUNT(*) FILTER (WHERE a.{quoted} IS DISTINCT FROM b.{quoted})")
    result = database.execute(
        """SELECT COUNT(*) FILTER (WHERE a.channel_id IS NULL OR b.channel_id IS NULL), """
        + ", ".join(conditions)
        + """ FROM read_parquet(?) a FULL OUTER JOIN read_parquet(?) b
              ON a.channel_id=b.channel_id AND a.prediction_time=b.prediction_time""",
        [str(left), str(right)],
    ).fetchone()
    differences = dict(zip(value_columns, result[1:], strict=True))
    if counts[0] != counts[1] or result[0] or any(differences.values()):
        raise ValueError(
            f"prediction content differs: rows={counts}, keys={result[0]}, values={differences}"
        )
    return {
        "rows_checked": counts[0],
        "key_mismatches": result[0],
        "value_mismatches": differences,
        "logical_schema_equal": True,
        "file_bytes_equal": sha256(left) == sha256(right),
    }


def run(
    *,
    freeze_path: Path,
    decision_path: Path,
    acceptance_path: Path,
    a_dir: Path,
    b_dir: Path,
    replay_path: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    freeze, decision, acceptance = (
        read_json(path) for path in (freeze_path, decision_path, acceptance_path)
    )
    freeze_hash = frozen_rule_sha256(freeze_path)
    if (
        decision["source_freeze_sha256"] != freeze_hash
        or acceptance["source_freeze_lf_sha256"] != freeze_hash
    ):
        raise ValueError("frozen rule fingerprint differs")
    a_manifest, a_report = verified_package(a_dir, acceptance["source_a_history_manifest_sha256"])
    b_manifest, b_report = verified_package(b_dir, decision["source_single_test_manifest_sha256"])
    if a_manifest["report_sha256"] != acceptance["source_a_history_report_sha256"]:
        raise ValueError("A raw-history report differs from acceptance")
    replay = read_json(replay_path)
    if (
        sha256(replay_path) != decision["source_saved_prediction_replay_report_sha256"]
        or replay["status"] != "passed"
        or replay["source_test_manifest_sha256"] != sha256(b_dir / "manifest.json")
        or replay["source_freeze_sha256"] != freeze_hash
    ):
        raise ValueError("saved A3 replay differs from pinned B result")
    if (
        b_manifest["source_freeze_sha256"] != freeze_hash
        or a_report["source_freeze_sha256"] != freeze_hash
        or b_report["source_freeze_sha256"] != freeze_hash
        or a_report["source_a3_manifest_sha256"] != freeze["source_a3_manifest_sha256"]
        or a_report["source_admission_manifest_sha256"]
        != freeze["source_r3_admission_manifest_sha256"]
    ):
        raise ValueError("raw-history or transferred package lineage differs")
    if (
        a_report["alerts"] != b_report["alerts"]
        or a_report["hourly_pr_auc"] != b_report["hourly_pr_auc"]
        or a_report["test_hours_checked"] != b_report["conditionally_admitted_hours"]
        or b_report["alerts"]["threshold"] != freeze["frozen_threshold"]
        or b_report["alerts"]["matched_episodes"] != decision["test_matched_episodes"]
        or b_report["alerts"]["unmatched_warnings"] != decision["test_unmatched_warnings"]
    ):
        raise ValueError("fixed-threshold metrics differ")
    a_months = {item["month"]: item for item in a_manifest["monthly_predictions"]}
    b_months = {item["month"]: item for item in b_manifest["monthly_predictions"]}
    monthly = []
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        database.execute("SET memory_limit='2GB'")
        for month in MONTHS:
            a_item, b_item = a_months[month], b_months[month]
            compared = compare_parquets(database, a_dir / a_item["file"], b_dir / b_item["file"])
            if compared["rows_checked"] != a_item["rows"] or a_item["rows"] != b_item["rows"]:
                raise ValueError(f"monthly row count differs: {month}")
            monthly.append({"month": month, **compared})
        alerts = compare_parquets(
            database, a_dir / "emitted_alerts.parquet", b_dir / "emitted_alerts.parquet"
        )
    rows = sum(item["rows_checked"] for item in monthly)
    if (
        rows != a_report["test_hours_checked"]
        or rows != replay["rows_checked"]
        or alerts["rows_checked"] != b_report["alerts"]["emitted_warnings"]
    ):
        raise ValueError("total prediction or warning count differs")
    report = {
        "schema_version": "r6-a-b-original-file-acceptance-v1",
        "status": "passed_exact_keyed_content_and_pinned_B_file_hashes",
        "source_freeze_lf_sha256": freeze_hash,
        "source_b_manifest_sha256": sha256(b_dir / "manifest.json"),
        "source_b_report_sha256": b_manifest["report_sha256"],
        "source_a_history_manifest_sha256": sha256(a_dir / "manifest.json"),
        "source_a_history_report_sha256": a_manifest["report_sha256"],
        "source_saved_prediction_replay_report_sha256": sha256(replay_path),
        "prediction_rows_checked": rows,
        "months": monthly,
        "emitted_alerts": alerts,
        "fixed_threshold_metrics_equal": True,
        "matched_episodes": b_report["alerts"]["matched_episodes"],
        "unmatched_warnings": b_report["alerts"]["unmatched_warnings"],
        "frozen_threshold": freeze["frozen_threshold"],
        "deployment_approved": False,
        "physical_failure_claim": False,
        "limitations": [
            "No model selection or threshold tuning on the already opened test.",
            "B file bytes match B pins; A/B content matches exactly by key, not necessarily file bytes.",
            "Past-only live admission and customer-approved warning budget remain unverified.",
            "The original A audit's missing-B note is historical; this report supplements it.",
        ],
    }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=Path("ml/r6_frozen_rule_v1.json"))
    parser.add_argument("--decision", type=Path, default=Path("ml/r6_b_final_decision_v1.json"))
    parser.add_argument(
        "--acceptance", type=Path, default=Path("ml/r6_a_technical_acceptance_v1.json")
    )
    parser.add_argument("--a-dir", type=Path, required=True)
    parser.add_argument("--b-dir", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        freeze_path=args.freeze,
        decision_path=args.decision,
        acceptance_path=args.acceptance,
        a_dir=args.a_dir,
        b_dir=args.b_dir,
        replay_path=args.replay_report,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
