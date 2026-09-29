"""Compare separate-process continuous and checkpoint-resumed archive runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from analysis.shadow_checkpoint_bundle import load_bundle
from analysis.train_r4_discrete_baselines import sha256
from ml.forecast.shadow_pilot import ShadowPolicy
from stage1.shadow.checkpoint import CheckpointError, canonical, parse_json


def run(
    *,
    continuous_dir: Path,
    resumed_dir: Path,
    output_dir: Path,
    freeze_path: Path,
    reference_dir: Path | None = None,
) -> dict:
    if output_dir.exists() or output_dir.with_name(output_dir.name + ".inprogress").exists():
        raise FileExistsError(output_dir)
    policy = ShadowPolicy.from_freeze(freeze_path)
    source = parse_json((continuous_dir / "checkpoint.json").read_bytes())["source_identity"]
    continuous = load_bundle(continuous_dir, source_identity=source, policy=policy)
    resumed = load_bundle(resumed_dir, source_identity=source, policy=policy)
    if continuous.next_prediction != continuous.end or resumed.next_prediction != resumed.end:
        raise CheckpointError("verification needs two complete runs of the pinned period")
    if (
        continuous.predictions != resumed.predictions
        or continuous.decisions != resumed.decisions
        or continuous.checkpoint() != resumed.checkpoint()
    ):
        raise CheckpointError("continuous and resumed decisions or final state differ")
    byte_equal = {
        name: sha256(continuous_dir / name) == sha256(resumed_dir / name)
        for name in (
            "checkpoint.json",
            "shadow_predictions.parquet",
            "hours.parquet",
            "shadow_decisions.parquet",
        )
    }
    if not all(byte_equal.values()):
        raise CheckpointError("continuous and resumed deterministic files differ")
    reference = None
    if reference_dir is not None:
        manifest = parse_json((reference_dir / "manifest.json").read_bytes())
        for name, key in (
            ("shadow_predictions.parquet", "prediction_sha256"),
            ("hours.parquet", "b_input_sha256"),
            ("report.json", "report_sha256"),
        ):
            if sha256(reference_dir / name) != manifest[key]:
                raise CheckpointError("accepted P1 reference file integrity differs")
        original_report = parse_json((reference_dir / "report.json").read_bytes())
        original = pq.ParquetFile(reference_dir / "shadow_predictions.parquet").read().to_pylist()
        if (
            original_report["source_m1_manifest_sha256"] != source["m1_manifest_sha256"]
            or original_report["source_freeze_lf_sha256"] != policy.freeze_sha256
            or original != continuous.predictions
        ):
            raise CheckpointError("restart implementation differs from the accepted P1 reference")
        reference = {
            "manifest_sha256": sha256(reference_dir / "manifest.json"),
            "rows_checked": len(original),
            "decision_mismatches": 0,
        }
    report = {
        "schema_version": "shadow-a-restart-verification-v1",
        "status": "technical_archive_restart_verified_b_review_pending",
        "continuous_manifest_sha256": sha256(continuous_dir / "manifest.json"),
        "resumed_manifest_sha256": sha256(resumed_dir / "manifest.json"),
        "source_identity": source,
        "start": continuous.start.isoformat(),
        "end_exclusive": continuous.end.isoformat(),
        "configured_channels": len(continuous.channels),
        "rows_checked": len(continuous.predictions),
        "conditionally_scored_hours": sum(
            row["rule_score"] is not None for row in continuous.predictions
        ),
        "shadow_warnings": sum(row["shadow_warning"] for row in continuous.decisions),
        "decision_mismatches": 0,
        "final_state_mismatches": 0,
        "deterministic_files_bitwise_equal": byte_equal,
        "final_source_cursor": continuous.checkpoint()["source_cursor"],
        "accepted_p1_reference": reference,
        "resources": {
            "continuous": parse_json((continuous_dir / "report.json").read_bytes())[
                "resources_this_process"
            ],
            "resumed": parse_json((resumed_dir / "report.json").read_bytes())[
                "resources_this_process"
            ],
        },
        "joint_pilot_acceptance": False,
        "live_source_contract_approved": False,
        "physical_failure_claim": False,
        "automatic_actions_enabled": False,
        "quality_metrics_computed": False,
        "limitations": [
            "Historical sorted archive only; actual delivery time and source watermark unavailable.",
            "No warnings on the fixed week; positive-warning restart is covered by synthetic tests.",
            "SHA-256 detects corruption; it is not authentication against an attacker rewriting files.",
            "Immutable directory commit covers process interruption, not a sudden power-loss guarantee.",
            "Bounded diagnostic snapshots include their complete output prefix; not a full-scale service.",
            "New live source, customer limits and independent B review remain pending.",
        ],
    }
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    pending.mkdir(parents=True)
    (pending / "report.json").write_bytes(canonical(report) + b"\n")
    (pending / "manifest.json").write_bytes(
        canonical(
            {
                "schema_version": report["schema_version"],
                "status": report["status"],
                "report_sha256": sha256(pending / "report.json"),
            }
        )
        + b"\n"
    )
    pending.rename(output_dir)
    return report


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-dir", type=Path, required=True)
    parser.add_argument("--resumed-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, default=Path("ml/r6_frozen_rule_v1.json"))
    args = parser.parse_args()
    report = run(
        continuous_dir=args.continuous_dir,
        resumed_dir=args.resumed_dir,
        output_dir=args.output_dir,
        freeze_path=args.freeze,
        reference_dir=args.reference_dir,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "rows_checked",
                    "decision_mismatches",
                    "final_state_mismatches",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
