"""Package the complete prepared training population, code and existing weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from ml.service_candidate.loader import ResearchRiskModel, sha256, source_sha256
from ml.service_candidate.train import B3_SHA, Q2_SHA, Q3_SHA, read_json


DATA_PARTS = ("train", "tune", "validation")
CODE_FILES = (
    "__init__.py",
    "features.py",
    "loader.py",
    "train.py",
    "training_bundle.py",
    "retrain_bundle.py",
    "test_service_candidate.py",
    "README.md",
    "requirements.txt",
)


def verify_bundle(directory: Path) -> dict:
    """Validate every extracted file before a reproducibility run."""
    directory = directory.resolve()
    manifest = read_json(directory / "bundle_manifest.json")
    if manifest["schema_version"] != "prepared-training-code-weights-v1":
        raise ValueError("unknown training bundle schema")
    for name, expected in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha256(path) != expected:
            raise ValueError(f"bundle file missing or changed: {name}")
    return manifest


def package_training_bundle(
    *,
    model: Path,
    data: Path,
    q2: Path,
    q3: Path,
    b3: Path,
    verification: Path,
    reference: Path,
    output: Path,
) -> dict:
    if output.exists() or output.with_name(output.name + ".inprogress").exists():
        raise FileExistsError(output)
    loaded = ResearchRiskModel(model)
    metadata = loaded.metadata
    data_manifest = read_json(data / "manifest.json")
    if (
        data_manifest["schema_version"] != "round2-q3-research-intake-v1"
        or data_manifest["train_years"] != [2019, 2020, 2022, 2023]
        or data_manifest["tune_year"] != 2024
        or data_manifest["validation_year"] != 2025
        or data_manifest["admission_policy"] != "combined"
        or data_manifest["physical_availability_approved"]
        or data_manifest["row_stats"]["train"]["rows"] != 1_011_240
        or metadata["source_train_manifest_sha256"] != sha256(data / "manifest.json")
        or metadata["source_q3_independent_verification_sha256"] != sha256(verification)
        or metadata["reference_model_sha256"] != sha256(reference / "engineered_episode.cbm")
    ):
        raise ValueError("model and prepared training population differ")
    for source, expected in ((q2, Q2_SHA), (q3, Q3_SHA), (b3, B3_SHA)):
        if sha256(source / "manifest.json") != expected:
            raise ValueError(f"source manifest differs: {source}")
    if read_json(reference / "selection.json")["selected_variant"] != "engineered_episode":
        raise ValueError("reference selection differs")
    source = Path(__file__).parent
    if source_sha256(source / "features.py") != metadata["feature_transform_source_sha256"]:
        raise ValueError("packaged feature transformation differs from model")
    included = {
        "model.cbm": model / "model.cbm",
        "model_metadata.json": model / "model_metadata.json",
        "training_report.json": model / "training_report.json",
        "data/manifest.json": data / "manifest.json",
        "source_manifests/q2/manifest.json": q2 / "manifest.json",
        "source_manifests/q3/manifest.json": q3 / "manifest.json",
        "source_manifests/b3/manifest.json": b3 / "manifest.json",
        "source_manifests/q3_verification.json": verification,
        "reference/engineered_episode.cbm": reference / "engineered_episode.cbm",
        "reference/selection.json": reference / "selection.json",
    }
    included.update({f"ml/service_candidate/{name}": source / name for name in CODE_FILES})
    for part in DATA_PARTS:
        path = data / f"{part}.parquet"
        if sha256(path) != data_manifest["files"][part]["sha256"]:
            raise ValueError(f"prepared {part} table differs from its manifest")
        included[f"data/{part}.parquet"] = path
    digests = {name: sha256(path) for name, path in included.items()}
    manifest = {
        "schema_version": "prepared-training-code-weights-v1",
        "description": "complete prepared train/tune/validation tables, retraining code, existing weights",
        "raw_260_million_event_sources_included": False,
        "train_rows": data_manifest["row_stats"]["train"]["rows"],
        "train_positive_hours": data_manifest["row_stats"]["train"]["positive_hours"],
        "train_years": data_manifest["train_years"],
        "tune_year": data_manifest["tune_year"],
        "validation_year_already_open": data_manifest["validation_year"],
        "excluded_year": 2021,
        "model_sha256": metadata["model_sha256"],
        "production_approved": False,
        "files_sha256": digests,
    }
    pending = output.with_name(output.name + ".inprogress")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = model.name
    with zipfile.ZipFile(
        pending, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3
    ) as archive:
        for name, path in sorted(included.items()):
            archive.write(
                path,
                f"{root}/{name}",
                compress_type=zipfile.ZIP_STORED
                if name.endswith(".parquet")
                else zipfile.ZIP_DEFLATED,
            )
        archive.writestr(f"{root}/bundle_manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(pending) as archive:
        if len(archive.namelist()) != len(included) + 1 or archive.testzip() is not None:
            raise ValueError("training ZIP failed integrity check")
        for name, expected in digests.items():
            import hashlib

            digest = hashlib.sha256()
            with archive.open(f"{root}/{name}") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"training ZIP member changed: {name}")
    pending.rename(output)
    return {
        "zip": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "members": len(included) + 1,
        "train_rows": manifest["train_rows"],
        "model_sha256": metadata["model_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "data", "q2", "q3", "b3", "verification", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(package_training_bundle(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == "__main__":
    main()
