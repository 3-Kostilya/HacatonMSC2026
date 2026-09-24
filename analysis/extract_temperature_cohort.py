"""Cache recent events for the temperature channels that passed profile filters."""

import json
from pathlib import Path
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/temperature_anomaly"
COLS = [
    "ид_события",
    "ид_канала_данных",
    "дата",
    "время",
    "тревожное",
    "значение_датчика",
]


def main():
    ids = set(
        json.loads((ROOT / "output/temperature_profile/audit.json").read_text(encoding="utf-8"))[
            "stable_channel_ids"
        ]
    )
    profile = pd.read_csv(
        ROOT / "output/temperature_profile/channel_year.csv", dtype={"channel_id": str}
    )
    frames = []
    for year in (2025, 2026):
        archive = ROOT / f"data/ext-journal-{year}.7z"
        prior = json.loads(
            (ROOT / f"analysis/results/ext-journal-{year}.json").read_text(encoding="utf-8")
        )
        if archive.stat().st_size != prior["bytes"]:
            raise ValueError("Archive changed since saved analysis")
        proc = subprocess.Popen(
            ["C:/Program Files/7-Zip/7z.exe", "x", "-so", str(archive)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        count = 0
        picked = []
        try:
            for chunk in pd.read_csv(
                proc.stdout,
                dtype=str,
                keep_default_na=False,
                usecols=COLS,
                chunksize=750_000,
                encoding="utf-8-sig",
            ):
                count += len(chunk)
                part = chunk.loc[chunk["ид_канала_данных"].isin(ids), COLS]
                if not part.empty:
                    picked.append(part.copy())
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            proc.stdout.close()
        error = proc.stderr.read().decode("utf-8", errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(error)
        frame = pd.concat(picked, ignore_index=True)
        expected = int(
            profile.loc[profile["year"].eq(year) & profile["channel_id"].isin(ids), "rows"].sum()
        )
        if count != prior["rows"] or len(frame) != expected:
            raise ValueError(f"Counts differ for {year}")
        frames.append(frame)
        print(f"{year}: {len(frame):,} events from {len(ids)} channels", flush=True)
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(OUT / "stable_temperature_events.parquet", index=False)


if __name__ == "__main__":
    main()
