"""Bounded M1-only proof of proposed sparse admission; not a full training index."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import zipfile

import duckdb
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_a2_hourly import _monthly_files
from analysis.r6_provenance import frozen_rule_sha256
from analysis.replay_shadow_pilot import COLUMNS, observation_groups, select_channels
from analysis.train_r4_discrete_baselines import read_json, sha256
from ml.forecast.r6_rule import TERMS
from stage1.features.sparse_admission import BLOCKING_QA, RELAXED_REASONS, VERSION
from stage1.features.sparse_admission import SparseAdmissionStream
from stage1.state_labeling.operational import ARCHIVE_SEGMENTS, segment_at
from stage1.state_labeling.rules import RULESET_VERSION
from stage1.value_quality import QA_VALUE_RULESET_VERSION


HOUR = timedelta(hours=1)
SCHEMA = pa.schema(
    [
        ("channel_id", pa.string()),
        ("prediction_time", pa.timestamp("us")),
        ("sensor_type", pa.string()),
        *[(name, pa.int64()) for name in TERMS],
        ("candidate_version", pa.string()),
        ("admission_status", pa.string()),
        ("admission_reasons", pa.list_(pa.string())),
        ("legacy_admission_status", pa.string()),
        ("legacy_admission_reasons", pa.list_(pa.string())),
        ("relaxed_data_reasons", pa.list_(pa.string())),
        ("blocking_qa_count_24h", pa.int64()),
        ("availability_status", pa.string()),
        ("last_explicit_normal_at", pa.timestamp("us")),
        ("history_through", pa.timestamp("us")),
        ("admission_through", pa.timestamp("us")),
        ("baseline_fit_end_at", pa.timestamp("us")),
    ]
)


def validate_bounds(start, end):
    if (
        any(
            at.tzinfo is not None or (at.minute, at.second, at.microsecond) != (0, 0, 0)
            for at in (start, end)
        )
        or end <= start
        or end - start > timedelta(days=31)
        or start.year not in {2019, 2020, 2022, 2023, 2024, 2025}
        or end > datetime(2026, 1, 1)
        or segment_at(start) is None
        or segment_at(start) != segment_at(end - HOUR)
    ):
        raise ValueError(
            "candidate replay needs train/validation whole hours in one segment, <=31 days"
        )


def replay(stream, groups, channels, start, end):
    validate_bounds(start, end)
    pending = iter(groups)
    group = next(pending, None)
    at = start
    while at < end:
        while group is not None and group[0].event.timestamp <= at:
            stream.observe_group(group)
            group = next(pending, None)
        yield stream.evaluate(at, channels)
        at += HOUR


def verify_decisions(path: Path, channels: list[str], start: datetime, end: datetime) -> dict:
    table = pq.read_table(path)
    if not table.schema.equals(SCHEMA, check_metadata=False):
        raise ValueError("candidate schema differs")
    rows = table.to_pylist()
    expected = {
        (channel, start + i * HOUR)
        for channel in channels
        for i in range(int((end - start) / HOUR))
    }
    keys = [(row["channel_id"], row["prediction_time"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("candidate grid incomplete or duplicated")
    counts, legacy, transitions, reasons, relaxed = (Counter() for _ in range(5))
    by_type = defaultdict(Counter)
    for row in rows:
        if row["candidate_version"] != VERSION or row["availability_status"] != "unknown":
            raise ValueError("candidate version or unverified availability changed")
        for name in ("history_through", "admission_through", "last_explicit_normal_at"):
            if row[name] is not None and row[name] > row["prediction_time"]:
                raise ValueError("future evidence in candidate output")
        status = row["admission_status"]
        old = row["legacy_admission_status"]
        if (
            old not in {"eligible", "unknown", "excluded"}
            or (old == "eligible" and row["legacy_admission_reasons"])
            or (old != "eligible" and not row["legacy_admission_reasons"])
            or row["blocking_qa_count_24h"] is None
            or row["blocking_qa_count_24h"] < 0
            or any(row[name] is None or row[name] < 0 for name in TERMS)
        ):
            raise ValueError("invalid legacy status, past counters or QA count")
        remaining = set(row["legacy_admission_reasons"]) - RELAXED_REASONS
        if row["blocking_qa_count_24h"]:
            remaining.add("qa_unusable_measurement_24h")
        expected_status = (
            "excluded" if old == "excluded" else "unknown" if remaining else "eligible"
        )
        if (
            row["admission_reasons"] != sorted(remaining)
            or status != expected_status
            or row["relaxed_data_reasons"]
            != sorted(set(row["legacy_admission_reasons"]) & RELAXED_REASONS)
            or (status == "eligible" and row["admission_reasons"])
        ):
            raise ValueError("candidate removed a non-statistical veto")
        counts[status] += 1
        legacy[old] += 1
        transitions[f"{old}->{status}"] += 1
        reasons.update(row["admission_reasons"])
        relaxed.update(row["relaxed_data_reasons"])
        kind = row["sensor_type"] or "<unknown_or_conflicting>"
        by_type[kind]["hours"] += 1
        by_type[kind]["legacy_eligible"] += int(old == "eligible")
        by_type[kind]["candidate_eligible"] += int(status == "eligible")
    return {
        "rows_checked": len(rows),
        "grid_mismatches": 0,
        "causal_timestamp_violations": 0,
        "non_statistical_veto_mismatches": 0,
        "legacy_status_counts": dict(legacy),
        "candidate_status_counts": dict(counts),
        "status_transitions": dict(transitions),
        "retained_reason_counts": dict(reasons),
        "relaxed_reason_counts": dict(relaxed),
        "by_type": {kind: dict(value) for kind, value in sorted(by_type.items())},
    }


def compare_saved_legacy(
    path: Path,
    legacy_dir: Path,
    *,
    m1_sha256: str,
    channels: list[str],
    start: datetime,
    end: datetime,
) -> dict:
    """Post-inference parity against the earlier accepted bounded replay."""
    manifest = read_json(legacy_dir / "manifest.json")
    report_file = legacy_dir / "report.json"
    report = read_json(report_file)
    old_path = (legacy_dir / manifest["prediction_file"]).resolve()
    if (
        legacy_dir.resolve() not in old_path.parents
        or manifest["report_sha256"] != sha256(report_file)
        or manifest["prediction_sha256"] != sha256(old_path)
        or report["source_m1_manifest_sha256"] != m1_sha256
        or report["start"] != start.isoformat()
        or report["end_exclusive"] != end.isoformat()
        or sorted(item["channel_id"] for item in report["selected_channels"]) != sorted(channels)
    ):
        raise ValueError("legacy replay lineage, scope or hash differs")
    old_rows = pq.read_table(old_path).to_pylist()
    old = {(row["channel_id"], row["prediction_time"]): row for row in old_rows}
    actual_rows = pq.read_table(path).to_pylist()
    if (
        len(old) != len(old_rows)
        or len(old_rows) != manifest["prediction_rows"]
        or {(row["channel_id"], row["prediction_time"]) for row in actual_rows} != set(old)
    ):
        raise ValueError("legacy replay keys differ or duplicate")
    for actual in actual_rows:
        reference = old[(actual["channel_id"], actual["prediction_time"])]
        for name in (
            *TERMS,
            "sensor_type",
            "availability_status",
            "last_explicit_normal_at",
            "history_through",
            "admission_through",
            "baseline_fit_end_at",
        ):
            if actual[name] != reference[name]:
                raise ValueError(f"legacy past evidence mismatch: {name}")
        if (
            actual["legacy_admission_status"] != reference["admission_status"]
            or actual["legacy_admission_reasons"] != reference["admission_reasons"]
        ):
            raise ValueError("legacy admission mismatch")
    return {
        "legacy_replay_manifest_sha256": sha256(legacy_dir / "manifest.json"),
        "legacy_rows_compared": len(actual_rows),
        "legacy_key_mismatches": 0,
        "legacy_four_counter_mismatches": 0,
        "legacy_past_guard_mismatches": 0,
    }


def run(
    *, m1_dir: Path, contract_path: Path, output_dir: Path, legacy_replay_dir: Path | None = None
) -> dict:
    begun = time.perf_counter()
    contract = read_json(contract_path)
    m1 = read_json(m1_dir / "manifest.json")
    if (
        contract["schema_version"] != VERSION
        or contract["joint_admission_approved"] is not False
        or contract["source_m1_manifest_sha256"] != sha256(m1_dir / "manifest.json")
        or set(contract["relaxed_reasons"]) != RELAXED_REASONS
        or set(contract["blocking_qa_categories_24h"]) != BLOCKING_QA
        or contract["ruleset_version"] != RULESET_VERSION
        or contract["qa_value_ruleset_version"] != QA_VALUE_RULESET_VERSION
        or m1["status"] != "complete"
    ):
        raise ValueError("candidate contract or M1 lineage differs")
    start = datetime.fromisoformat(contract["diagnostic_start"])
    end = datetime.fromisoformat(contract["diagnostic_end_exclusive"])
    # Validate bounds before opening any M1 event file.
    validate_bounds(start, end)
    maximum = contract["diagnostic_max_channels"]
    if not isinstance(maximum, int) or not 1 <= maximum <= 20:
        raise ValueError("candidate diagnostic supports 1..20 pre-start-selected channels")
    history_start = ARCHIVE_SEGMENTS[segment_at(start)][0]
    files, missing = _monthly_files(m1_dir, history_start, end)
    if missing or not files or any("year=2026" in path.as_posix() for path in files):
        raise ValueError(f"candidate history files missing or test file present: {missing}")
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or pending.exists():
        raise FileExistsError(output_dir)
    sources = [
        {
            "file": path.relative_to(m1_dir).as_posix(),
            "sha256": sha256(path),
            "metadata_rows": pq.ParquetFile(path).metadata.num_rows,
        }
        for path in files
    ]
    pending.mkdir(parents=True)
    output_file = pending / "candidate_admission.parquet"
    stream = SparseAdmissionStream()
    with duckdb.connect() as database:
        database.execute("SET threads=2")
        database.execute("SET memory_limit='2GB'")
        selected = select_channels(database, files, history_start, start, maximum)
        channels = [row["channel_id"] for row in selected]
        if not channels:
            raise ValueError("no pre-start channels available")
        print(f"selected {len(channels)} channels from past only", flush=True)
        database.execute(
            "CREATE TEMP TABLE selected AS SELECT UNNEST(?::VARCHAR[]) channel_id", [channels]
        )
        reader = database.execute(
            "SELECT " + ",".join(COLUMNS) + " FROM read_parquet(?,hive_partitioning=false) "
            "SEMI JOIN selected USING(channel_id) WHERE timestamp>=? AND timestamp<? "
            "AND split_part(replace(source,chr(92),'/'),'/',-1)="
            "'ext-journal-' || CAST(year(timestamp) AS VARCHAR) || '.7z' "
            "ORDER BY timestamp,channel_id,row_id",
            [[str(path) for path in files], history_start, end],
        ).to_arrow_reader(batch_size=25_000)
        rows = (row for batch in reader for row in batch.to_pylist())
        with pq.ParquetWriter(output_file, SCHEMA, compression="zstd") as writer:
            for decisions in replay(stream, observation_groups(rows), channels, start, end):
                writer.write_table(pa.Table.from_pylist(decisions, schema=SCHEMA))
    verification = verify_decisions(output_file, channels, start, end)
    if legacy_replay_dir is not None:
        verification.update(
            compare_saved_legacy(
                output_file,
                legacy_replay_dir,
                m1_sha256=sha256(m1_dir / "manifest.json"),
                channels=channels,
                start=start,
                end=end,
            )
        )
    memory = psutil.Process().memory_info()
    report = {
        "schema_version": VERSION,
        "status": "bounded_causal_candidate_proof_b_review_pending",
        "source_m1_manifest_sha256": sha256(m1_dir / "manifest.json"),
        "source_contract_lf_sha256": frozen_rule_sha256(contract_path),
        "code_lf_sha256": {
            name: frozen_rule_sha256(Path(__file__).resolve().parents[1] / name)
            for name in (
                "analysis/replay_sparse_admission_a.py",
                "analysis/replay_shadow_pilot.py",
                "analysis/build_a2_hourly.py",
                "stage1/features/sparse_admission.py",
                "stage1/shadow/stream.py",
                "stage1/features/hourly.py",
                "stage1/features/r2.py",
                "stage1/state_labeling/registered_episodes.py",
                "stage1/state_labeling/operational.py",
                "stage1/state_labeling/rules.py",
                "stage1/value_quality.py",
            )
        },
        "sources": sources,
        "selected_channels": selected,
        "history_start": history_start.isoformat(),
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "accepted_observations": stream.accepted_rows,
        "selection_uses_only_pre_start_observations": True,
        "inference_reads_only_m1": True,
        "future_labels_read": False,
        "new_training_index_created": False,
        "joint_admission_approved": False,
        "scores_or_warnings_generated": False,
        "physical_failure_claim": False,
        "full_train_validation_coverage_computed": False,
        "precision_recall_computed": False,
        **verification,
        "resources": {
            "elapsed_seconds": round(time.perf_counter() - begun, 3),
            "peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss),
            "memory_measurement": "Windows process peak; RSS fallback elsewhere",
        },
        "limitations": [
            "Bounded pre-start-selected diagnostic channels, not full admission or episode coverage.",
            "Four past counters exported for parity only, not the full missing-aware feature pack.",
            "Baseline recomputation still uses the old daily causal tracker; not performance optimized.",
            "Recent normal and archive completeness are journal assumptions, not proof of channel uptime.",
            "New QA veto, relaxed rich-statistics requirements and missing-aware model need joint B review.",
            "No training, threshold tuning, new test, notifications or deployment approval.",
        ],
    }
    report_file = pending / "report.json"
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": VERSION,
        "status": report["status"],
        "decision_rows": verification["rows_checked"],
        "decision_file": output_file.name,
        "decision_sha256": sha256(output_file),
        "report_sha256": sha256(report_file),
    }
    (pending / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pending.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, required=True)
    parser.add_argument(
        "--contract", type=Path, default=Path("ml/sparse_admission_proposal_v1.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-replay-dir", type=Path)
    parser.add_argument("--handoff-zip", type=Path)
    args = parser.parse_args()
    result = run(
        m1_dir=args.m1_dir,
        contract_path=args.contract,
        output_dir=args.output_dir,
        legacy_replay_dir=args.legacy_replay_dir,
    )
    if args.handoff_zip:
        print(json.dumps(create_handoff(args.output_dir, args.handoff_zip)), flush=True)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "rows_checked",
                    "legacy_status_counts",
                    "candidate_status_counts",
                    "status_transitions",
                    "resources",
                )
            }
        )
    )


def create_handoff(output_dir: Path, zip_path: Path) -> dict:
    """Package verified outputs only; preserve the folder and forbid overwrites."""
    manifest = read_json(output_dir / "manifest.json")
    if manifest["decision_file"] != "candidate_admission.parquet":
        raise ValueError("unexpected decision file in candidate manifest")
    files = [
        output_dir / name
        for name in ("manifest.json", "report.json", "candidate_admission.parquet")
    ]
    expected = {file.name: sha256(file) for file in files}
    if (
        expected["report.json"] != manifest["report_sha256"]
        or expected["candidate_admission.parquet"] != manifest["decision_sha256"]
    ):
        raise ValueError("candidate output hash differs before packaging")
    pending = zip_path.with_name(zip_path.name + ".inprogress")
    if zip_path.exists() or pending.exists():
        raise FileExistsError(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(pending, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, f"{output_dir.name}/{file.name}")
    import hashlib

    with zipfile.ZipFile(pending) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(files):
            raise ValueError("candidate ZIP integrity failed")
        for file in files:
            saved = archive.read(f"{output_dir.name}/{file.name}")
            if hashlib.sha256(saved).hexdigest() != expected[file.name]:
                raise ValueError("candidate ZIP member hash differs")
    pending.rename(zip_path)
    return {
        "handoff_zip": str(zip_path),
        "bytes": zip_path.stat().st_size,
        "sha256": sha256(zip_path),
        "member_hashes_verified": len(files),
    }


if __name__ == "__main__":
    main()
