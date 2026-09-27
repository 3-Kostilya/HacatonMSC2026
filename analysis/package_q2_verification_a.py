"""Package small Q2/A verification artifacts, never models or full feature files."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from analysis.build_sparse_population_a import write_json
from analysis.package_sparse_population_a import package_handoff
from analysis.train_r4_discrete_baselines import read_json, sha256
from analysis.verify_q2_b_metrics_a import DECISION_SHA, GAS_SHA


FOLDERS = (
    "q2-a-oracle-capacity-20260927",
    "q2-a-b-score-verification-20260927",
    "q2-a-b-metrics-acceptance-20260927",
    "q2-a-b-errors-replay-20260927",
    "q2-a-b-refined-thresholds-replay-20260927",
)
SINGLE_FILES = {
    "q2-a-b-final-thresholds-replay-20260927.json": DECISION_SHA,
    "q2-a-b-gas-burst-replay-20260927.json": GAS_SHA,
}


def verified_members(folder: Path) -> dict[str, str]:
    manifest_path = folder / "manifest.json"
    manifest = read_json(manifest_path)
    if "files" in manifest:
        members = {name: info["sha256"] for name, info in manifest["files"].items()}
    else:
        mapping = {
            "report_sha256": "report.json",
            "curves_sha256": "curves.json",
            "warning_cases_sha256": "warning_cases.parquet",
            "episode_cases_sha256": "episode_cases.parquet",
        }
        members = {name: manifest[key] for key, name in mapping.items() if key in manifest}
    if "report.json" not in members:
        raise ValueError("verification package has no hashed report")
    for name, digest in members.items():
        file = (folder / name).resolve()
        if folder.resolve() not in file.parents or sha256(file) != digest:
            raise ValueError("verification member path/hash differs")
    return {"manifest.json": sha256(manifest_path), **members}


def package(*, output_root: Path, output_dir: Path, output: Path) -> dict:
    pending = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or pending.exists() or output.exists():
        raise FileExistsError("verification handoff destination already exists")
    sources = {}
    for name in FOLDERS:
        folder = output_root / name
        for member, digest in verified_members(folder).items():
            sources[f"{name}/{member}"] = (folder / member, digest)
    for name, digest in SINGLE_FILES.items():
        source = output_root / name
        if sha256(source) != digest:
            raise ValueError("published B replay hash differs")
        sources[name] = (source, digest)
    acceptance = read_json(output_root / FOLDERS[2] / "report.json")
    if (
        acceptance["status"] != "technical_negative_experiment_accepted_by_A"
        or acceptance["requirements_met"]
        or acceptance["all_score_verification_manifest_sha256"]
        != sha256(output_root / FOLDERS[1] / "manifest.json")
    ):
        raise ValueError("A acceptance is incomplete or has changed lineage")
    pending.mkdir(parents=True)
    members = {}
    for name, (source, digest) in sources.items():
        destination = pending / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256(destination) != digest:
            raise ValueError("handoff copy changed")
        members[name] = {"sha256": digest, "bytes": destination.stat().st_size}
    write_json(
        pending / "manifest.json",
        {
            "schema_version": "q2-a-small-verification-handoff-v1",
            "files": members,
            "months": [],
            "contains_future_label_oracle_do_not_use_as_features": True,
            "contains_no_raw_events_models_or_full_feature_files": True,
            "requirements_met": False,
        },
    )
    pending.rename(output_dir)
    return package_handoff(output_dir, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-root", "output-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(package(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
