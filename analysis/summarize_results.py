import collections
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "results"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize():
    years = [read(OUT / f"ext-journal-{y}.json") for y in range(2019, 2027)]
    sample = read(OUT / "журнал_событий_пример.json")
    totals = {"rows": sum(s["rows"] for s in years)}
    fields = [
        "alarm",
        "values",
        "daily",
        "daily_alarm",
        "daily_failure",
        "channels",
        "failure_channels",
        "alarm_channels",
        "types",
        "systems",
        "failure_types",
        "failure_systems",
        "failure_alarm",
        "missing",
    ]
    for field in fields:
        c = collections.Counter()
        for s in years:
            c.update(s[field])
        totals[field] = dict(c)
    for field in [
        "numeric_rows",
        "negative_numeric_rows",
        "unknown_rows",
        "invalid_date_rows",
        "invalid_time_rows",
        "duplicate_event_ids",
        "duplicate_full_rows_hash64",
        "adjacent_time_reversals",
        "failure_channel_days",
    ]:
        totals[field] = sum(s[field] for s in years)
    overlaps = []
    sampleids = np.load(OUT / "журнал_событий_пример.ids.npy")
    for i, s in enumerate(years):
        y = 2019 + i
        valid_dates = {}
        invalid_dates = {}
        for d, n in s["daily"].items():
            try:
                dt.date.fromisoformat(d)
                valid_dates[d] = n
            except ValueError:
                invalid_dates[d] = n
        s["invalid_date_values"] = invalid_dates
        invalid_channels = [k for k in s["channels"] if not k.isdecimal()]
        s["invalid_channel_markers"] = invalid_channels
        s["distinct_channels"] -= len(invalid_channels)
        s["unknown_channels"] -= len(invalid_channels)
        days = sorted(valid_dates)
        s["date_min"], s["date_max"] = days[0], days[-1]
        valid = {dt.date.fromisoformat(d) for d in days}
        lo, hi = min(valid), max(valid)
        s["days_observed"] = len(valid)
        s["span_days"] = (hi - lo).days + 1
        s["missing_days_within_span"] = [
            (lo + dt.timedelta(days=k)).isoformat()
            for k in range(s["span_days"])
            if lo + dt.timedelta(days=k) not in valid
        ]
        s["out_of_filename_year_rows"] = sum(
            n for d, n in valid_dates.items() if not d.startswith(str(y))
        )
        s["days_top"] = sorted(valid_dates.items(), key=lambda x: -x[1])[:5]
        s["days_bottom"] = sorted(valid_dates.items(), key=lambda x: x[1])[:5]
        s["channel_top10_share"] = (
            sum(sorted(s["channels"].values(), reverse=True)[:10]) / s["rows"]
        )
        arr = np.load(OUT / f"ext-journal-{y}.ids.npy", mmap_mode="r")
        pos = np.searchsorted(arr, sampleids)
        positions = np.minimum(pos, len(arr) - 1)
        hit = (pos < len(arr)) & (arr[positions] == sampleids)
        s["sample_id_overlap"] = int(hit.sum())
        for j in range(i):
            prev = years[j]
            if (
                s["min_event_id"] <= prev["max_event_id"]
                and prev["min_event_id"] <= s["max_event_id"]
            ):
                other = np.load(OUT / f"ext-journal-{2019 + j}.ids.npy", mmap_mode="r")
                low = max(s["min_event_id"], prev["min_event_id"])
                high = min(s["max_event_id"], prev["max_event_id"])
                sub = other[
                    np.searchsorted(other, low) : np.searchsorted(other, high, side="right")
                ]
                count = 0
                for start in range(0, len(sub), 1000000):
                    batch = sub[start : start + 1000000]
                    at = np.searchsorted(arr, batch)
                    count += int(
                        ((at < len(arr)) & (arr[np.minimum(at, len(arr) - 1)] == batch)).sum()
                    )
                overlaps.append({"years": [2019 + j, y], "matching_id_rows_from_earlier": count})
        del arr
    totals["cross_year_id_overlaps"] = overlaps
    totals["sample_id_overlap"] = sum(s["sample_id_overlap"] for s in years)
    totals["monthly"] = {}
    for field in ["daily", "daily_alarm", "daily_failure"]:
        counter = collections.Counter()
        for d, n in totals[field].items():
            try:
                dt.date.fromisoformat(d)
                counter[d[:7]] += n
            except ValueError:
                pass
        totals["monthly"][field] = dict(sorted(counter.items()))
    value_alarm = collections.Counter()
    for s in years:
        for val, a, n in s["value_alarm"]:
            value_alarm[(val, a)] += n
    totals["value_alarm"] = [[val, a, n] for (val, a), n in value_alarm.most_common()]
    result = {
        "years": years,
        "sample": sample,
        "totals": totals,
        "dictionaries": read(OUT / "dictionaries.json"),
    }
    (OUT / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = [
        "file",
        "rows",
        "date_min",
        "date_max",
        "days_observed",
        "distinct_channels",
        "unknown_channels",
        "unknown_rows",
        "numeric_rows",
        "duplicate_event_ids",
        "duplicate_full_rows_hash64",
        "sample_id_overlap",
    ]
    print(pd.DataFrame(years)[columns].to_string(index=False))
    print("TOTAL_ROWS", totals["rows"])
    print("FAILURE_TYPES", totals["failure_types"])
    print("TOP_VALUES", sorted(totals["values"].items(), key=lambda x: -x[1])[:35])
    print("CROSS_YEAR_OVERLAPS", overlaps)


if __name__ == "__main__":
    summarize()
