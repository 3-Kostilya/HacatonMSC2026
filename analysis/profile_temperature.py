"""Profile temperature channels in 2024-2026 using the existing journal method."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/temperature_profile"
YEARS = (2024, 2025, 2026)
COLS = ["ид_канала_данных", "дата", "время", "значение_датчика", "тревожное"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dictionary_path = ROOT / "data/справочник_каналов_датчиков.csv"
    dictionary = pd.read_csv(dictionary_path, dtype=str, keep_default_na=False)
    selected = dictionary.loc[dictionary["тип_датчика"].eq("Датчик температуры")]
    ids = set(selected["ид_канала_данных"])
    audit = json.loads((ROOT / "output/dictionary_check/audit.json").read_text(encoding="utf-8"))
    if hashlib.sha256(dictionary_path.read_bytes()).hexdigest() != audit["channels"]["sha256"]:
        raise ValueError("Dictionary changed since the previous audit")

    parts = []
    scanned = {}
    for year in YEARS:
        archive = ROOT / f"data/ext-journal-{year}.7z"
        previous = json.loads(
            (ROOT / f"analysis/results/ext-journal-{year}.json").read_text(encoding="utf-8")
        )
        if archive.stat().st_size != previous["bytes"]:
            raise ValueError(f"Archive changed since prior analysis: {archive.name}")
        proc = subprocess.Popen(
            ["C:/Program Files/7-Zip/7z.exe", "x", "-so", str(archive)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        count = 0
        retained = 0
        started = time.monotonic()
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
                picked = chunk.loc[chunk["ид_канала_данных"].isin(ids)].copy()
                retained += len(picked)
                if not picked.empty:
                    parts.append(picked)
                if count % 7_500_000 < 750_000:
                    print(
                        f"{year}: {count:,} rows read in {time.monotonic() - started:.0f}s",
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
        if count != previous["rows"] or retained != previous["types"]["Датчик температуры"]:
            raise ValueError(f"Counts differ from prior analysis for {year}")
        scanned[str(year)] = {"archive_rows": count, "temperature_rows": retained}
        print(f"{year}: retained {retained:,} temperature rows", flush=True)

    df = pd.concat(parts, ignore_index=True)[COLS]
    df.columns = ["channel_id", "date", "time", "value", "alarm"]
    df["year"] = df["date"].str[:4].astype(int)
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df["number"] = pd.to_numeric(df["value"], errors="coerce")
    df.loc[~np.isfinite(df["number"]), "number"] = np.nan
    df["numeric"] = df["number"].notna()
    df["review_range"] = df["numeric"] & ((df["number"] < -50) | (df["number"] > 100))
    df["valid_alarm"] = df["alarm"].isin(["t", "f", "true", "false"])

    duplicate_semantic = int(df.duplicated(["channel_id", "timestamp", "value", "alarm"]).sum())
    conflicts = int(
        df.groupby(["channel_id", "timestamp"], dropna=False)["value"].nunique().gt(1).sum()
    )
    raw = df.groupby(["channel_id", "year"]).agg(
        rows=("value", "size"),
        numeric_rows=("numeric", "sum"),
        review_range_rows=("review_range", "sum"),
        alarm_rows=("alarm", lambda x: x.isin(["t", "true"]).sum()),
    )
    numeric = df.loc[df["numeric"] & df["timestamp"].notna()].copy()
    numeric = numeric.drop_duplicates(["channel_id", "timestamp", "number"])
    numeric = numeric.sort_values(["channel_id", "timestamp"])
    numeric["gap_hours"] = (
        numeric.groupby("channel_id")["timestamp"].diff().dt.total_seconds() / 3600
    )
    numeric["day"] = numeric["timestamp"].dt.floor("D")
    per_numeric = numeric.groupby(["channel_id", "year"]).agg(
        distinct_numeric_points=("timestamp", "size"),
        numeric_days=("day", "nunique"),
        median_gap_hours=("gap_hours", "median"),
        p90_gap_hours=("gap_hours", lambda x: x.quantile(0.90)),
        max_gap_hours=("gap_hours", "max"),
        min_number=("number", "min"),
        max_number=("number", "max"),
    )
    channel_year = raw.join(per_numeric, how="left").reset_index()
    channel_year["numeric_share"] = channel_year["numeric_rows"] / channel_year["rows"]
    channel_year.to_csv(OUT / "channel_year.csv", index=False, encoding="utf-8-sig")

    year_summary = []
    for year in YEARS:
        group = df.loc[df["year"].eq(year)]
        numeric_group = numeric.loc[numeric["year"].eq(year)]
        expected_numeric = json.loads(
            (ROOT / f"analysis/results/ext-journal-{year}.json").read_text(encoding="utf-8")
        )["numeric_by_type"]["Датчик температуры"]["count"]
        if int(group["numeric"].sum()) != expected_numeric:
            raise ValueError(f"Numeric temperature count differs from previous analysis: {year}")
        year_summary.append(
            {
                "year": year,
                "rows": len(group),
                "channels_with_events": int(group["channel_id"].nunique()),
                "numeric_rows": int(group["numeric"].sum()),
                "channels_with_numeric": int(group.loc[group["numeric"], "channel_id"].nunique()),
                "text_rows": int((~group["numeric"]).sum()),
                "review_range_rows": int(group["review_range"].sum()),
                "numeric_days_median_per_channel": float(
                    channel_year.loc[channel_year["year"].eq(year), "numeric_days"].median()
                ),
                "numeric_gap_p50_hours_all_points": float(numeric_group["gap_hours"].quantile(0.5)),
                "numeric_gap_p90_hours_all_points": float(numeric_group["gap_hours"].quantile(0.9)),
                "top_text_values": group.loc[~group["numeric"], "value"]
                .value_counts()
                .head(12)
                .to_dict(),
                "top_review_values": group.loc[group["review_range"], "value"]
                .value_counts()
                .head(12)
                .to_dict(),
            }
        )

    recent = channel_year.pivot(index="channel_id", columns="year", values="numeric_rows").fillna(0)
    numeric_100_50 = recent[2025].ge(100) & recent[2026].ge(50)
    days = channel_year.pivot(index="channel_id", columns="year", values="numeric_days").fillna(0)
    days_30_15 = days[2025].ge(30) & days[2026].ge(15)
    p90 = channel_year.pivot(index="channel_id", columns="year", values="p90_gap_hours")
    gaps_24 = p90[2025].le(24) & p90[2026].le(24)
    stable = numeric_100_50 & days_30_15 & gaps_24
    result = {
        "method": "Full scan of 2024-2026 journals, filtered to current temperature channel IDs",
        "sources": scanned,
        "rows": len(df),
        "invalid_timestamps": int(df["timestamp"].isna().sum()),
        "invalid_alarm_codes": int((~df["valid_alarm"]).sum()),
        "duplicate_semantic_rows": duplicate_semantic,
        "conflicting_channel_timestamps": conflicts,
        "year_summary": year_summary,
        "channels_numeric_100_in_2025_and_50_in_2026": int(numeric_100_50.sum()),
        "channels_numeric_days_30_in_2025_and_15_in_2026": int(days_30_15.sum()),
        "channels_p90_gap_at_most_24h_both_years": int(gaps_24.sum()),
        "channels_meeting_all_three_filters": int(stable.sum()),
        "stable_channel_ids": sorted(recent.index[stable].tolist()),
    }
    (OUT / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "stable_channel_ids"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
