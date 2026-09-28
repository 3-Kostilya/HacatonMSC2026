"""Create a hash-checked ZIP containing research model weights and full code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from ml.service_candidate.loader import ResearchRiskModel, sha256, source_sha256


def package_model(model: Path, destination: Path) -> dict:
    if not model.is_dir() or destination.exists():
        raise FileExistsError("model source missing or package destination already exists")
    pending = destination.with_name(destination.name + ".inprogress")
    if pending.exists():
        raise FileExistsError(pending)
    ResearchRiskModel(model)
    source = Path(__file__).parent
    root = model.name
    included = {
        f"{root}/{name}": model / name
        for name in ("model.cbm", "model_metadata.json", "training_report.json")
    }
    included.update(
        {
            f"{root}/ml/service_candidate/{name}": source / name
            for name in (
                "__init__.py",
                "features.py",
                "loader.py",
                "train.py",
                "package.py",
                "test_service_candidate.py",
                "README.md",
                "requirements.txt",
            )
        }
    )
    metadata = json.loads((model / "model_metadata.json").read_text(encoding="utf-8"))
    if source_sha256(source / "features.py") != metadata["feature_transform_source_sha256"]:
        raise ValueError("training feature source changed before packaging")
    digests = {name: sha256(path) for name, path in included.items()}
    manifest = {
        "schema_version": "service-research-model-bundle-v1",
        "files_sha256": digests,
        "production_approved": False,
        "automatic_actions_allowed": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(pending, "x", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name, path in sorted(included.items()):
            archive.write(path, name)
        archive.writestr(f"{root}/package_manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(pending) as archive:
        if len(archive.namelist()) != len(included) + 1 or archive.testzip() is not None:
            raise ValueError("incomplete ZIP or failed CRC")
        import hashlib

        for name, expected in digests.items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"ZIP member checksum differs: {name}")
    pending.rename(destination)
    return {
        "zip": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "members": len(included) + 1,
        "model_sha256": metadata["model_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_model(args.model, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
