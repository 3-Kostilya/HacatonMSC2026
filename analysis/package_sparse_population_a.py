"""Verify and package a full Q2 A handoff, excluding temporary database spill files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile

from analysis.train_r4_discrete_baselines import read_json, sha256


def package_handoff(package: Path, destination: Path) -> dict:
    manifest = read_json(package / "manifest.json")
    members = {"manifest.json": sha256(package / "manifest.json")}
    members.update({name: info["sha256"] for name, info in manifest["files"].items()})
    for month in manifest["months"]:
        relative = Path(month["manifest_file"])
        members[relative.as_posix()] = month["manifest_sha256"]
        for name, info in month["files"].items():
            members[(relative.parent / name).as_posix()] = info["sha256"]
    for relative, expected in members.items():
        resolved = (package / relative).resolve()
        if package.resolve() not in resolved.parents or sha256(resolved) != expected:
            raise ValueError("full package member hash or path differs")
    pending = destination.with_name(destination.name + ".inprogress")
    if destination.exists() or pending.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        pending, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3
    ) as archive:
        for relative in sorted(members):
            archive.write(package / relative, f"{package.name}/{relative}")
    with zipfile.ZipFile(pending) as archive:
        if len(archive.namelist()) != len(members):
            raise ValueError("full handoff ZIP member count differs")
        for relative, expected in members.items():
            digest = hashlib.sha256()
            with archive.open(f"{package.name}/{relative}") as item:
                for block in iter(lambda: item.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError("full handoff ZIP CRC or member SHA differs")
    pending.rename(destination)
    return {
        "file": str(destination),
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
        "verified_members": len(members),
        "source_manifest_sha256": members["manifest.json"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(package_handoff(args.package, args.output))


if __name__ == "__main__":
    main()
