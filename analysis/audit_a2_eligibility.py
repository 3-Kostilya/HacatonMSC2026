"""Diagnose the existing A2 QA slice without changing its eligibility decisions.

The numeric and discrete counts below describe available *past data*, not
approved training rows. Future labels and model-specific admission require the
new R1 target contract and are intentionally left uncomputed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.features import A2_SCHEMA, FEATURE_VERSION, validate_a2_table  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _numeric_profile_data_candidate(row: dict[str, Any]) -> bool:
    """Both past numeric descriptions exist; this is not model eligibility."""

    return (
        row["sensor_type"] is not None
        and row["baseline_status"] == "eligible"
        and row["baseline_numeric_count"] > 0
        and row["baseline_numeric_median"] is not None
        and row["baseline_numeric_mad"] is not None
        and row["numeric_count_24h"] > 0
        and row["numeric_median_24h"] is not None
        and row["excluded_quality_count_24h"] == 0
        and "insufficient_history" not in row["availability_reasons"]
    )


def _discrete_history_data_candidate(row: dict[str, Any]) -> bool:
    """Past state observations are technically available, not yet interpreted."""

    return (
        row["sensor_type"] is not None
        and row["baseline_status"] == "eligible"
        and row["baseline_state_count"] > 0
        and row["state_count_24h"] > 0
        and row["state_transitions_24h"] is not None
        and row["excluded_quality_count_24h"] == 0
        and "same_time_state_ambiguity" not in row["window_reasons_24h"]
        and "insufficient_history" not in row["availability_reasons"]
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count overlapping original blockers and separate data-availability branches."""

    total = len(rows)
    availability_reasons = Counter(
        reason for row in rows for reason in set(row["availability_reasons"])
    )
    reason_combinations = Counter(tuple(sorted(set(row["availability_reasons"]))) for row in rows)
    numeric_candidates = [_numeric_profile_data_candidate(row) for row in rows]
    state_candidates = [_discrete_history_data_candidate(row) for row in rows]
    channels: dict[str, list[tuple[dict[str, Any], bool, bool]]] = defaultdict(list)
    for row, numeric, state in zip(rows, numeric_candidates, state_candidates, strict=True):
        channels[row["channel_id"]].append((row, numeric, state))

    channel_diagnostics = []
    for channel_id, members in sorted(channels.items()):
        channel_rows = [item[0] for item in members]
        channel_diagnostics.append(
            {
                "channel_id": channel_id,
                "sensor_types": sorted(
                    {str(row["sensor_type"]) for row in channel_rows if row["sensor_type"]}
                ),
                "rows": len(members),
                "original_availability_counts": _counts(
                    [row["availability_status"] for row in channel_rows]
                ),
                "original_reason_counts": _counts(
                    [reason for row in channel_rows for reason in set(row["availability_reasons"])]
                ),
                "numeric_profile_data_candidates": sum(item[1] for item in members),
                "discrete_history_data_candidates": sum(item[2] for item in members),
                "discrete_candidates_without_24h_numeric": sum(
                    item[2] and item[0]["numeric_count_24h"] == 0 for item in members
                ),
            }
        )

    return {
        "status": "diagnostic_only",
        "row_count": total,
        "channel_count": len(channels),
        "original_availability": {
            "status_counts": _counts([row["availability_status"] for row in rows]),
            "reason_counts_overlapping": dict(sorted(availability_reasons.items())),
            "universal_reasons": sorted(
                reason for reason, count in availability_reasons.items() if count == total
            ),
            "reason_combinations": [
                {"reasons": list(reasons), "rows": count}
                for reasons, count in sorted(
                    reason_combinations.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "note": "Reasons overlap and their counts must not be summed as distinct rows.",
        },
        "baseline": {
            "status_counts": _counts([row["baseline_status"] for row in rows]),
            "reason_counts_overlapping": _counts(
                [reason for row in rows for reason in set(row["baseline_reasons"])]
            ),
        },
        "windows": {
            f"{hours}h": {
                "status_counts": _counts([row[f"window_status_{hours}h"] for row in rows]),
                "reason_counts_overlapping": _counts(
                    [reason for row in rows for reason in set(row[f"window_reasons_{hours}h"])]
                ),
            }
            for hours in (1, 6, 24, 168)
        },
        "numeric_profile": {
            "numeric_observed_24h_rows": sum(row["numeric_count_24h"] > 0 for row in rows),
            "numeric_missing_24h_rows": sum(row["numeric_count_24h"] == 0 for row in rows),
            "numeric_observed_in_baseline_rows": sum(
                row["baseline_numeric_count"] > 0 for row in rows
            ),
            "data_candidate_rows": sum(numeric_candidates),
            "data_candidate_channels": len(
                {
                    row["channel_id"]
                    for row, candidate in zip(rows, numeric_candidates, strict=True)
                    if candidate
                }
            ),
            "data_candidates_with_positive_mad": sum(
                candidate and row["baseline_numeric_mad"] > 0
                for row, candidate in zip(rows, numeric_candidates, strict=True)
            ),
            "candidate_definition": (
                "Known type; eligible past baseline with numeric median/MAD; 24h numeric "
                "median; no 24h quality exclusions; sufficient original causal history. "
                "This only measures feature data presence."
            ),
        },
        "discrete_history": {
            "state_observed_24h_rows": sum(row["state_count_24h"] > 0 for row in rows),
            "state_observed_in_baseline_rows": sum(row["baseline_state_count"] > 0 for row in rows),
            "data_candidate_rows": sum(state_candidates),
            "data_candidate_channels": len(
                {
                    row["channel_id"]
                    for row, candidate in zip(rows, state_candidates, strict=True)
                    if candidate
                }
            ),
            "data_candidates_without_24h_numeric": sum(
                candidate and row["numeric_count_24h"] == 0
                for row, candidate in zip(rows, state_candidates, strict=True)
            ),
            "data_candidates_without_any_baseline_or_24h_numeric": sum(
                candidate and row["numeric_count_24h"] == 0 and row["baseline_numeric_count"] == 0
                for row, candidate in zip(rows, state_candidates, strict=True)
            ),
            "candidate_definition": (
                "Known type; eligible past baseline with state observations; 24h state "
                "observations without same-time ambiguity or quality exclusions; sufficient "
                "original causal history. Unknown cadence is not treated as known coverage. "
                "State semantics are not established by this count."
            ),
        },
        "model_usability": {
            "status": "not_yet_approved",
            "eligible_rows": None,
            "reason": "Requires jointly accepted R1 rules and a model-specific feature/eligibility contract.",
        },
        "future_label_eligibility": {
            "status": "not_computable_until_b_target",
            "eligible_rows": None,
            "reason": "Requires B's target definition, state episodes and future observability rules.",
        },
        "channel_diagnostics": channel_diagnostics,
    }


def audit(a2_dir: Path, *, max_rows: int = 100_000) -> dict[str, Any]:
    """Verify one bounded A2 artifact, then return a non-mutating diagnostic."""

    a2_dir = a2_dir.resolve()
    manifest_path = a2_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("schema_version") != FEATURE_VERSION:
        raise ValueError("A2 manifest must be complete with the expected feature version")
    if manifest.get("features_file") != "features.parquet":
        raise ValueError("A2 manifest must name features.parquet")
    features_path = a2_dir / "features.parquet"
    metadata = pq.ParquetFile(features_path).metadata
    if metadata.num_rows > max_rows:
        raise ValueError(f"A2 diagnostic exceeds {max_rows} row safety limit")
    if metadata.num_rows != manifest.get("feature_rows"):
        raise ValueError("A2 manifest and Parquet row counts differ")
    features_sha = _sha256(features_path)
    if features_sha != manifest.get("features_sha256"):
        raise ValueError("A2 features SHA-256 differs from manifest")
    table = pq.read_table(features_path)
    if not table.schema.equals(A2_SCHEMA, check_metadata=False):
        raise ValueError("A2 Parquet schema differs from declared version")
    validate_a2_table(table)
    rows = table.to_pylist()
    for row in rows:
        if any(
            row[field] != manifest.get(field)
            for field in ("run_id", "config_sha256", "input_manifest_sha256")
        ):
            raise ValueError("A2 row provenance differs from manifest")
    validation_path = a2_dir / "validation_report.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        actual_counts = _counts([row["availability_status"] for row in rows])
        if validation.get("availability_counts") != actual_counts:
            raise ValueError("A2 validation report and Parquet availability counts differ")

    report = summarize_rows(rows)
    report["source"] = {
        "a2_dir": str(a2_dir),
        "a2_manifest_sha256": _sha256(manifest_path),
        "features_sha256": features_sha,
        "run_id": manifest["run_id"],
        "config_sha256": manifest["config_sha256"],
        "input_manifest_sha256": manifest["input_manifest_sha256"],
        "selection_mode": manifest.get("config", {}).get("selection_mode"),
        "selection_uses_target_presence_for_qa_only": manifest.get("config", {}).get(
            "selection_mode"
        )
        == "seeded_type_stratified_real_20_v1",
        "start_at": manifest.get("config", {}).get("start_at"),
        "end_at": manifest.get("config", {}).get("end_at"),
    }
    report["limitations"] = [
        "The saved 20-channel slice was chosen for structural QA using interval presence; "
        "it is not a representative train/test population.",
        "A2 cadence was not specified, so original unknown window status cannot establish "
        "future or even past interval completeness.",
        "Data-candidate counts do not override original A2 statuses and do not approve "
        "numeric or state models.",
    ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a2_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Write a new JSON report; never overwrite")
    parser.add_argument("--max-rows", type=int, default=100_000)
    args = parser.parse_args()
    if args.max_rows < 1:
        parser.error("--max-rows must be positive")
    report = audit(args.a2_dir, max_rows=args.max_rows)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
