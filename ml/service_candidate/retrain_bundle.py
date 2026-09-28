"""Retrain the unchanged research CatBoost directly from an extracted full bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.service_candidate.train import run
from ml.service_candidate.training_bundle import verify_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.bundle.resolve()
    verify_bundle(directory)
    output = args.output or directory / "retrained_model"
    report = run(
        data=directory / "data",
        q2=directory / "source_manifests" / "q2",
        q3=directory / "source_manifests" / "q3",
        b3=directory / "source_manifests" / "b3",
        verification=directory / "source_manifests" / "q3_verification.json",
        reference=directory / "reference",
        output=output,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
