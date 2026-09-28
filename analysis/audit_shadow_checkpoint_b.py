"""Independent B audit of A's published and locally repeated restart packages."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from analysis.shadow_checkpoint_bundle import load_bundle
from analysis.train_r4_discrete_baselines import sha256
from ml.forecast.shadow_pilot import ShadowPolicy, ShadowState, run_shadow_batch
from stage1.shadow.checkpoint import CheckpointError, canonical, parse_json


DATA_FILES = (
    "checkpoint.json",
    "hours.parquet",
    "shadow_decisions.parquet",
    "shadow_predictions.parquet",
)


def _load(directory: Path, policy: ShadowPolicy):
    manifest = parse_json((directory / "manifest.json").read_bytes())
    payload = parse_json((directory / "checkpoint.json").read_bytes())
    if set(manifest["files_sha256"]) != {*DATA_FILES, "report.json"}:
        raise CheckpointError("package does not declare every required file")
    for name, digest in manifest["files_sha256"].items():
        if sha256(directory / name) != digest:
            raise CheckpointError(f"package hash differs: {name}")
    session = load_bundle(directory, source_identity=payload["source_identity"], policy=policy)
    inputs = pq.read_table(directory / "hours.parquet").to_pylist()
    if len(inputs) != len(session.decisions):
        raise CheckpointError("B input and decision counts differ")
    return manifest, session, inputs


def _reject_rehashed_cooldown_change(directory: Path, policy: ShadowPolicy) -> None:
    with TemporaryDirectory() as temporary:
        changed = Path(temporary) / "changed"
        shutil.copytree(directory, changed)
        checkpoint_path = changed / "checkpoint.json"
        payload = parse_json(checkpoint_path.read_bytes())
        channel = payload["channels"][0]
        payload["b_state"]["last_warning_at"][channel] = payload["source_cursor"][
            "closed_through"
        ]
        checkpoint_path.write_bytes(canonical(payload) + b"\n")
        manifest_path = changed / "manifest.json"
        manifest = parse_json(manifest_path.read_bytes())
        manifest["files_sha256"]["checkpoint.json"] = sha256(checkpoint_path)
        manifest_path.write_bytes(canonical(manifest) + b"\n")
        try:
            load_bundle(changed, source_identity=payload["source_identity"], policy=policy)
        except CheckpointError:
            return
        raise AssertionError("rehashed B cooldown tampering was accepted")


def audit(*, a_continuous: Path, a_prefix: Path, a_resumed: Path,
          b_continuous: Path, b_prefix: Path, b_resumed: Path,
          reference: Path, freeze: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    policy = ShadowPolicy.from_freeze(freeze)
    packages = {
        "a_continuous": a_continuous,
        "a_prefix": a_prefix,
        "a_resumed": a_resumed,
        "b_continuous": b_continuous,
        "b_prefix": b_prefix,
        "b_resumed": b_resumed,
    }
    loaded = {name: _load(directory, policy) for name, directory in packages.items()}
    original_manifest = parse_json((reference / "manifest.json").read_bytes())
    original_predictions = pq.read_table(reference / "shadow_predictions.parquet").to_pylist()
    if sha256(reference / "shadow_predictions.parquet") != original_manifest["prediction_sha256"]:
        raise CheckpointError("accepted P1 reference integrity differs")
    full = loaded["b_continuous"][1]
    prefix = loaded["b_prefix"][1]
    if len(full.predictions) != 3360 or len(prefix.predictions) != 1680:
        raise CheckpointError("fixed review slice has unexpected row count")
    if full.predictions != original_predictions:
        raise CheckpointError("new replay differs from accepted P1 rows")
    for name in ("a_continuous", "a_resumed", "b_resumed"):
        session = loaded[name][1]
        if (session.predictions != full.predictions
                or session.decisions != full.decisions
                or session.checkpoint() != full.checkpoint()):
            raise CheckpointError(f"final state or decisions differ: {name}")
    if (loaded["a_prefix"][1].checkpoint() != prefix.checkpoint()
            or prefix.predictions != full.predictions[:1680]
            or prefix.decisions != full.decisions[:1680]):
        raise CheckpointError("prefix differs from the continuous run")
    deterministic_equal = {}
    for name in DATA_FILES:
        digests = {sha256(directory / name) for directory in packages.values()
                   if directory not in (a_prefix, b_prefix)}
        deterministic_equal[name] = len(digests) == 1
    if not all(deterministic_equal.values()):
        raise CheckpointError("full-package deterministic files differ")
    recomputed_state = ShadowState()
    recomputed = run_shadow_batch(loaded["b_continuous"][2], recomputed_state, policy)
    if (recomputed != full.decisions
            or recomputed_state.checkpoint(policy) != full.b.checkpoint(policy)):
        raise CheckpointError("B decisions or cooldown cannot be reproduced from saved input")
    if any(row["automatic_action_taken"] or row["delivery_mode"] != "record_only"
           for row in full.decisions):
        raise CheckpointError("package contains an external action")
    _reject_rehashed_cooldown_change(b_prefix, policy)
    report = {
        "schema_version": "shadow-b-independent-restart-audit-v1",
        "status": "b_technical_archive_restart_accepted",
        "package_manifest_sha256": {
            name: sha256(directory / "manifest.json") for name, directory in packages.items()
        },
        "accepted_p1_manifest_sha256": sha256(reference / "manifest.json"),
        "rows_checked": len(full.decisions),
        "prefix_rows_checked": len(prefix.decisions),
        "scored_hours": sum(row["prediction_status"] == "scored" for row in full.decisions),
        "shadow_warnings": sum(row["shadow_warning"] for row in full.decisions),
        "b_decision_mismatches": 0,
        "final_state_mismatches": 0,
        "deterministic_files_bitwise_equal": deterministic_equal,
        "rehashed_b_cooldown_tampering_rejected": True,
        "live_source_accepted": False,
        "customer_limits_approved": False,
        "external_actions_enabled": False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_bytes(canonical(report) + b"\n")
    return report


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("a-continuous", "a-prefix", "a-resumed", "b-continuous",
                 "b-prefix", "b-resumed", "reference", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--freeze", type=Path, default=Path("ml/r6_frozen_rule_v1.json"))
    report = audit(**vars(parser.parse_args()))
    print(json.dumps({key: report[key] for key in (
        "status", "rows_checked", "prefix_rows_checked", "scored_hours",
        "shadow_warnings", "rehashed_b_cooldown_tampering_rejected",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
