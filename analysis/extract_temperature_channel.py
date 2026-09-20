"""Extract a selected temperature channel from 2025-2026 annual journals."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/temperature_anomaly"
COLUMNS = [
    "ид_события",
    "ид_канала_данных",
    "дата",
    "время",
    "тревожное",
    "значение_датчика",
]


def extract(channel_id):
    audit = json.loads((ROOT / "output/dictionary_check/audit.json").read_text(encoding="utf-8"))
    dictionary = ROOT / "dataset/справочник_каналов_датчиков.csv"
    if hashlib.sha256(dictionary.read_bytes()).hexdigest() != audit["channels"]["sha256"]:
        raise ValueError("Dictionary changed since its audit")
    channels = pd.read_csv(dictionary, dtype=str, keep_default_na=False)
    selected = channels.loc[channels["ид_канала_данных"].eq(channel_id)]
    if len(selected) != 1 or selected.iloc[0]["тип_датчика"] != "Датчик температуры":
        raise ValueError("Channel is not a unique temperature channel")
    profile = pd.read_csv(
        ROOT / "output/temperature_profile/channel_year.csv",
        dtype={"channel_id": str},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    provenance = {"channel_id": channel_id, "dictionary_sha256": audit["channels"]["sha256"]}
    for year in (2025, 2026):
        prior = json.loads(
            (ROOT / f"analysis/results/ext-journal-{year}.json").read_text(encoding="utf-8")
        )
        archive = ROOT / f"dataset/ext-journal-{year}.7z"
        if archive.stat().st_size != prior["bytes"]:
            raise ValueError(f"Archive changed: {archive.name}")
        proc = subprocess.Popen(
            ["C:/Program Files/7-Zip/7z.exe", "x", "-so", str(archive)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        year_frames = []
        scanned = 0
        try:
            for chunk in pd.read_csv(
                proc.stdout,
                dtype=str,
                keep_default_na=False,
                usecols=COLUMNS,
                chunksize=750_000,
                encoding="utf-8-sig",
            ):
                scanned += len(chunk)
                selected_chunk = chunk.loc[chunk["ид_канала_данных"].eq(channel_id), COLUMNS]
                if not selected_chunk.empty:
                    year_frames.append(selected_chunk.copy())
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            proc.stdout.close()
        error = proc.stderr.read().decode("utf-8", errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(error)
        frame = pd.concat(year_frames, ignore_index=True)
        expected = int(
            profile.loc[
                profile["channel_id"].eq(channel_id) & profile["year"].eq(year), "rows"
            ].iloc[0]
        )
        if scanned != prior["rows"] or len(frame) != expected:
            raise ValueError(
                f"Unexpected count for {year}: {scanned} scanned, {len(frame)} selected"
            )
        frames.append(frame)
        provenance[str(year)] = {"archive_bytes": prior["bytes"], "rows": len(frame)}
        print(f"{year}: selected {len(frame):,} rows", flush=True)
    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values(["дата", "время", "ид_события"])
    events.to_csv(OUT / f"events_{channel_id}.csv", index=False, encoding="utf-8-sig")
    (OUT / f"events_{channel_id}.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-id", default="2943")
    args = parser.parse_args()
    extract(args.channel_id)
