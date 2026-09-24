"""Publish a bounded R2 state-history supplement for an existing A2 QA slice."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.audit_a2_eligibility import audit as audit_a2  # noqa: E402
from analysis.build_a2_hourly import _events_for_channel, _monthly_files  # noqa: E402
from stage1.features.r2 import R2_VERSION, R2_STATE_SCHEMA, build_state_history_rows  # noqa: E402
from stage1.state_labeling.rules import RULESET_VERSION  # noqa: E402


MAX_CHANNELS = 20
MAX_DAYS = 31


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(*, a2_dir: Path, m1_manifest: Path, output: Path) -> dict[str, Any]:
    """Preserve the A2 slice and add only past semantic/availability columns."""

    a2_dir = a2_dir.resolve()
    m1_manifest = m1_manifest.resolve()
    output = output.resolve()
    pending = output.with_name(output.name + ".inprogress")
    if output.exists() or pending.exists():
        raise FileExistsError("R2 output or .inprogress directory already exists")
    a2_audit = audit_a2(a2_dir)
    a2_manifest_path = a2_dir / "manifest.json"
    a2_manifest = json.loads(a2_manifest_path.read_text(encoding="utf-8"))
    config = a2_manifest["config"]
    channels = config["channels"]
    if not 1 <= len(channels) <= MAX_CHANNELS:
        raise ValueError("R2 A supports only the bounded A2 channel slice")
    start_at = datetime.fromisoformat(config["start_at"])
    end_at = datetime.fromisoformat(config["end_at"])
    if (
        start_at.tzinfo is not None
        or end_at.tzinfo is not None
        or not start_at < end_at <= start_at + timedelta(days=MAX_DAYS)
    ):
        raise ValueError("R2 A requires a bounded local-time A2 slice")
    m1 = json.loads(m1_manifest.read_text(encoding="utf-8"))
    if m1.get("status") != "complete" or m1.get("scope") != "full_supplied_sources":
        raise ValueError("R2 A requires a complete full M1 manifest")
    m1_sha = _sha256(m1_manifest)
    if m1_sha != a2_manifest["input_manifest_sha256"]:
        raise ValueError("A2 was built from a different M1 manifest")
    a2_rows = pq.read_table(a2_dir / "features.parquet").to_pylist()
    by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in a2_rows:
        if row["channel_id"] not in channels:
            raise ValueError("A2 row channel is absent from its manifest")
        by_channel[row["channel_id"]].append(row)
    if set(by_channel) != set(channels):
        raise ValueError("A2 rows do not cover every selected channel")
    context_start = start_at - timedelta(hours=168)
    files, missing_months = _monthly_files(m1_manifest.parent, context_start, end_at)
    if missing_months or not files:
        raise ValueError(f"R2 A semantic history has missing source months: {missing_months}")
    parquet = ds.dataset(files, format="parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending.mkdir()
    feature_path = pending / "state_history.parquet"
    status_counts: dict[str, Counter[str]] = {
        name: Counter() for name in ("numeric_data_status", "discrete_data_status")
    }
    message_totals: Counter[str] = Counter()
    by_channel_report: list[dict[str, Any]] = []
    by_type_status: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {"numeric_data_status": Counter(), "discrete_data_status": Counter()}
    )
    row_count = 0
    writer = pq.ParquetWriter(feature_path, R2_STATE_SCHEMA, compression="zstd")
    try:
        for channel_id in sorted(channels):
            events = _events_for_channel(parquet, channel_id, context_start, end_at)
            table = build_state_history_rows(
                by_channel[channel_id],
                events,
                source_a2_manifest_sha256=_sha256(a2_manifest_path),
            )
            writer.write_table(table)
            row_count += table.num_rows
            channel_status = {
                name: Counter() for name in ("numeric_data_status", "discrete_data_status")
            }
            for row in table.to_pylist():
                for name, counts in status_counts.items():
                    counts[row[name]] += 1
                    channel_status[name][row[name]] += 1
                    by_type_status[channels[channel_id] or "<unknown>"][name][row[name]] += 1
                for name in (
                    "technical_message_count_24h",
                    "registered_fault_text_count_24h",
                    "unknown_state_count_24h",
                ):
                    message_totals[name] += row[name]
            by_channel_report.append(
                {
                    "channel_id": channel_id,
                    "sensor_type": channels[channel_id],
                    "rows": table.num_rows,
                    "numeric_data_status": dict(
                        sorted(channel_status["numeric_data_status"].items())
                    ),
                    "discrete_data_status": dict(
                        sorted(channel_status["discrete_data_status"].items())
                    ),
                }
            )
    finally:
        writer.close()
    if row_count != a2_manifest["feature_rows"]:
        raise ValueError("R2 supplement row count differs from A2")
    if pq.ParquetFile(feature_path).metadata.num_rows != row_count:
        raise ValueError("R2 physical Parquet row count differs from A2")
    report = {
        "schema_version": R2_VERSION,
        "status": "conditional_r2_a_qa",
        "source_a2_manifest_sha256": _sha256(a2_manifest_path),
        "source_a2_features_sha256": a2_manifest["features_sha256"],
        "source_m1_manifest_sha256": m1_sha,
        "ruleset_version": RULESET_VERSION,
        "row_count": row_count,
        "channel_count": len(channels),
        "original_a2_availability": a2_audit["original_availability"]["status_counts"],
        "operation_data_status": {
            name: dict(sorted(counts.items())) for name, counts in status_counts.items()
        },
        "by_type_operation_data_status": {
            sensor_type: {name: dict(sorted(counts.items())) for name, counts in status.items()}
            for sensor_type, status in sorted(by_type_status.items())
        },
        "by_channel_operation_data_status": by_channel_report,
        "semantic_24h_counts_summed_over_hours": dict(sorted(message_totals.items())),
        "episode_history_status": "catalog_unavailable",
        "model_admission_status": "unknown_until_model_contract",
        "future_label_status": "unknown_computed_by_b",
        "qa_selection_uses_target_period_presence": a2_audit["source"][
            "selection_uses_target_presence_for_qa_only"
        ],
        "limitations": [
            "R2 rows supplement the same bounded 20-channel A2 QA slice, not a train/test population.",
            "Semantic counts use only messages at or before prediction_time.",
            "Completed episode features remain null until B publishes an accepted catalog.",
            "Numeric and discrete statuses describe past data; model and future-label admission remain unknown.",
            "Archive completeness is a conditional assumption, not proven channel continuity.",
        ],
    }
    report_path = pending / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": R2_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_a2_manifest": str(a2_manifest_path),
        "source_a2_manifest_sha256": _sha256(a2_manifest_path),
        "source_m1_manifest": str(m1_manifest),
        "source_m1_manifest_sha256": m1_sha,
        "ruleset_version": RULESET_VERSION,
        "row_count": row_count,
        "files": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (feature_path, report_path)
        },
    }
    (pending / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a2-dir", required=True, type=Path)
    parser.add_argument("--m1-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(a2_dir=args.a2_dir, m1_manifest=args.m1_manifest, output=args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
