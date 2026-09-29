"""Resumable month-wise B3 labeling of A3's complete causal population."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_r1_state_mapping import _sha256  # noqa: E402
from analysis.build_r3_full_registered_labels import build_month  # noqa: E402


def _completed(output_root: Path, chunk: dict, a3_sha: str) -> bool:
    month = chunk["month"]
    directory = output_root / f"year={month[:4]}" / f"month={month[5:]}"
    if not directory.exists():
        return False
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete_month"
        or manifest.get("month") != month
        or manifest.get("source_a3_full_manifest_sha256") != a3_sha
        or manifest.get("source_a3_month_manifest_sha256") != chunk["manifest_sha256"]
        or manifest.get("row_count") != chunk["rows"]
    ):
        raise ValueError(f"existing B3 month is incomplete or differs: {month}")
    for name, meta in manifest["files"].items():
        if _sha256(directory / name) != meta["sha256"]:
            raise ValueError(f"existing B3 month file differs: {month}/{name}")
    return True


def run(*, a3_dir: Path, m1_manifest: Path, b2_dir: Path,
        output_root: Path, workers: int) -> None:
    if not 1 <= workers <= 3:
        raise ValueError("workers must be between 1 and 3")
    a3_dir, m1_manifest, b2_dir, output_root = (
        path.resolve() for path in (a3_dir, m1_manifest, b2_dir, output_root)
    )
    a3_path = a3_dir / "manifest.json"
    a3 = json.loads(a3_path.read_text(encoding="utf-8"))
    a3_sha = _sha256(a3_path)
    remaining = [
        chunk["month"] for chunk in a3["chunks"]
        if not _completed(output_root, chunk, a3_sha)
    ]
    print(json.dumps({"months_total": len(a3["chunks"]), "months_remaining": len(remaining),
                      "workers": workers}), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                build_month, a3_dir=a3_dir, m1_manifest=m1_manifest,
                b2_dir=b2_dir, month=month, output_root=output_root,
            ): month for month in remaining
        }
        for future in as_completed(futures):
            month = futures[future]
            report = future.result()
            print(json.dumps({
                "month": month, "rows": report["row_count"],
                "labels": report["label_status"], "elapsed_seconds": report["elapsed_seconds"],
            }, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--b2-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
