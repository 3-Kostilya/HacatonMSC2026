"""Stream annual archives and retain smoke-sensor events only."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "smoke_failure"
CHANNEL = "ид_канала_данных"
SENSOR_TYPE = "тип_датчика"
SMOKE_TYPE = "Датчик дыма"
FAULT = "Неисправен"


def find_seven_zip():
    """Resolve 7-Zip without assuming a particular operating system."""
    configured = os.environ.get("SEVEN_ZIP")
    candidates = [configured] if configured else ["7zz", "7z", r"C:/Program Files/7-Zip/7z.exe"]
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError(
        "7-Zip not found. Install 7z/7zz or set SEVEN_ZIP to its executable path."
    )


def source_signature(years=(2022, 2023)):
    sources = {}
    for year in years:
        path = ROOT / "data" / f"ext-journal-{year}.7z"
        stat = path.stat()
        sources[path.name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    dictionary = ROOT / "data" / "справочник_каналов_датчиков.csv"
    sources[dictionary.name] = {"sha256": hashlib.sha256(dictionary.read_bytes()).hexdigest()}
    return sources


def extract(years=(2022, 2023), output=OUT):
    output.mkdir(parents=True, exist_ok=True)
    dictionary = pd.read_csv(
        ROOT / "data" / "справочник_каналов_датчиков.csv",
        dtype=str,
        keep_default_na=False,
    )
    if dictionary[CHANNEL].duplicated().any():
        raise ValueError("Channel dictionary must have unique IDs")
    selected = dictionary.loc[dictionary[SENSOR_TYPE].eq(SMOKE_TYPE)].copy()
    ids = set(selected[CHANNEL])
    signature = source_signature(years)
    frames, audit = [], {}
    for year in years:
        cache = output / f"smoke_events_{year}.pkl"
        cache_meta = cache.with_suffix(".json")
        provenance = {
            "archive": signature[f"ext-journal-{year}.7z"],
            "dictionary": signature["справочник_каналов_датчиков.csv"],
            "sensor_type": SMOKE_TYPE,
        }
        previous = json.loads(cache_meta.read_text(encoding="utf-8")) if cache_meta.exists() else {}
        if not cache.exists() and output != OUT:
            shared = OUT / cache.name
            shared_meta = shared.with_suffix(".json")
            shared_info = (
                json.loads(shared_meta.read_text(encoding="utf-8")) if shared_meta.exists() else {}
            )
            if shared.exists() and shared_info.get("provenance") == provenance:
                frames.append(pd.read_pickle(shared))
                audit[str(year)] = shared_info["counts"]
                print(f"{year}: reused verified smoke-event cache", flush=True)
                continue
        if cache.exists() and previous.get("provenance") == provenance:
            frames.append(pd.read_pickle(cache))
            audit[str(year)] = previous["counts"]
            continue
        archive = ROOT / "data" / f"ext-journal-{year}.7z"
        proc = subprocess.Popen(
            [find_seven_zip(), "x", "-so", str(archive)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        parts, count, start = [], 0, time.monotonic()
        try:
            for chunk in pd.read_csv(
                proc.stdout,
                dtype=str,
                keep_default_na=False,
                chunksize=500_000,
                encoding="utf-8-sig",
            ):
                parts.append(chunk.loc[chunk[CHANNEL].isin(ids)].copy())
                count += len(chunk)
                if count % 5_000_000 == 0:
                    print(
                        f"{year}: scanned {count:,} rows in {time.monotonic() - start:.0f}s",
                        flush=True,
                    )
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            proc.stdout.close()
        error = proc.stderr.read().decode("utf-8", errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(error)
        frame = pd.concat(parts, ignore_index=True)
        frame.to_pickle(cache)
        frames.append(frame)
        audit[str(year)] = {"archive_rows": count, "smoke_rows": len(frame)}
        cache_meta.write_text(
            json.dumps(
                {"provenance": provenance, "counts": audit[str(year)]}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        print(f"{year}: retained {len(frame):,} smoke events", flush=True)
    events = pd.concat(frames, ignore_index=True)
    events = events.rename(
        columns={
            CHANNEL: "channel_id",
            "ид_события": "event_id",
            "значение_датчика": "state",
            "тревожное": "alarm",
        }
    )
    events["timestamp"] = pd.to_datetime(events["дата"] + " " + events["время"], errors="raise")
    if not events["alarm"].isin(["t", "f", "true", "false"]).all():
        raise ValueError("Unexpected alarm encoding")
    events["alarm"] = events["alarm"].isin(["t", "true"])
    raw_rows = len(events)
    # Event IDs are not globally unique; semantic duplicates are redundant for state histories.
    events = events.drop_duplicates(["channel_id", "timestamp", "state", "alarm"])
    events = events.sort_values(["channel_id", "timestamp", "state", "alarm"]).reset_index(
        drop=True
    )
    events = events[["channel_id", "timestamp", "state", "alarm"]]
    events.to_pickle(output / "events.pkl")
    events.to_parquet(output / "events.parquet", index=False)
    selected.rename(columns={CHANNEL: "channel_id"}).to_csv(
        output / "smoke_channels.csv", index=False
    )
    gaps = events.groupby("channel_id")["timestamp"].diff().dt.total_seconds() / 3600
    conflicts = events.groupby(["channel_id", "timestamp"])["state"].nunique()
    audit.update(
        {
            "sources": signature,
            "raw_smoke_rows": raw_rows,
            "events_after_deduplication": len(events),
            "channels": events["channel_id"].nunique(),
            "states": events["state"].value_counts().to_dict(),
            "conflicting_timestamps": int(conflicts.gt(1).sum()),
            "gap_hours_quantiles": gaps.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict(),
            "states_by_year": {
                str(y): g["state"].value_counts().to_dict()
                for y, g in events.groupby(events.timestamp.dt.year)
            },
        }
    )
    (output / "extraction_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return events


if __name__ == "__main__":
    extract()
