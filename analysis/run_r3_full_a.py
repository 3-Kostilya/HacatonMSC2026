"""Resume all full-population R3 A months with bounded local parallelism."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from analysis.finalize_r3_full_pack import _months, finalize


def _run_month(
    month: str, *, m1_manifest: Path, population_dir: Path,
    b2_dir: Path, output_root: Path,
) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "analysis.build_r3_full_month",
        "--m1-manifest", str(m1_manifest),
        "--population-dir", str(population_dir),
        "--b2-dir", str(b2_dir),
        "--month", month,
        "--output-root", str(output_root),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"{month}: {completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout.splitlines()[-1])


def run(*, m1_manifest: Path, population_dir: Path, b2_dir: Path,
        output_root: Path, workers: int = 4) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("R3 full-run worker count must be 1-8")
    m1_manifest, population_dir, b2_dir, output_root = (
        path.resolve() for path in (m1_manifest, population_dir, b2_dir, output_root)
    )
    remaining = []
    for year, month in _months():
        name = f"{year}-{month:02d}"
        directory = output_root / f"year={year}" / f"month={month:02d}"
        pending = directory.with_name(directory.name + ".inprogress")
        if pending.exists():
            print(f"skip in-progress {name}; it must finish before finalization", flush=True)
            continue
        if directory.exists():
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("status") != "complete_month" or manifest.get("month") != name:
                raise ValueError(f"existing R3 month is not complete: {name}")
            print(f"skip completed {name}", flush=True)
        else:
            remaining.append(name)
    print(f"R3 full A: {len(remaining)} remaining months, {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = iter(remaining)
        active = {}

        def submit_next() -> None:
            try:
                month = next(iterator)
            except StopIteration:
                return
            future = pool.submit(
                _run_month, month, m1_manifest=m1_manifest,
                population_dir=population_dir, b2_dir=b2_dir, output_root=output_root,
            )
            active[future] = month

        for _ in range(min(workers, len(remaining))):
            submit_next()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                month = active.pop(future)
                result = future.result()
                print(
                    f"complete {month}: {result['row_count']} hours, "
                    f"{result['elapsed_seconds']}s",
                    flush=True,
                )
                submit_next()
    return finalize(
        output_root=output_root,
        population_dir=population_dir,
        m1_manifest=m1_manifest,
        b2_manifest=b2_dir / "manifest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-manifest", required=True, type=Path)
    parser.add_argument("--population-dir", required=True, type=Path)
    parser.add_argument("--b2-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = run(
        m1_manifest=args.m1_manifest,
        population_dir=args.population_dir,
        b2_dir=args.b2_dir,
        output_root=args.output_root,
        workers=args.workers,
    )
    print(json.dumps({key: manifest[key] for key in (
        "status", "row_count", "chunk_count", "feature_count"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
