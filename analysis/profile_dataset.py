import collections
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "results"
OUT.mkdir(parents=True, exist_ok=True)
SEVEN = r"C:/Program Files/7-Zip/7z.exe"
COLS = ["ид_события", "ид_канала_данных", "дата", "время", "тревожное", "значение_датчика"]
dictionary = pd.read_csv(
    ROOT / "data" / "справочник_каналов_датчиков.csv", dtype=str, keep_default_na=False
)
type_map = dictionary.drop_duplicates(COLS[1]).set_index(COLS[1])["тип_датчика"].to_dict()
system_map = dictionary.drop_duplicates(COLS[1]).set_index(COLS[1])["тип_инж_системы"].to_dict()
known = set(type_map)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def profile(path):
    start = time.time()
    label = path.stem
    archive = path.suffix == ".7z"
    stats = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "rows": 0,
        "columns": COLS,
        "missing": collections.Counter(),
        "alarm": collections.Counter(),
        "values": collections.Counter(),
        "daily": collections.Counter(),
        "daily_alarm": collections.Counter(),
        "daily_failure": collections.Counter(),
        "channels": collections.Counter(),
        "failure_channels": collections.Counter(),
        "alarm_channels": collections.Counter(),
        "types": collections.Counter(),
        "systems": collections.Counter(),
        "failure_types": collections.Counter(),
        "failure_systems": collections.Counter(),
        "failure_alarm": collections.Counter(),
        "text_alarm": collections.Counter(),
        "unknown_rows": 0,
        "numeric_rows": 0,
        "invalid_date_rows": 0,
        "invalid_time_rows": 0,
        "negative_numeric_rows": 0,
        "adjacent_time_reversals": 0,
        "non_integer_ids": 0,
        "examples": [],
        "failure_examples": [],
        "unknown_examples": [],
    }
    if archive:
        listing = subprocess.check_output([SEVEN, "l", "-slt", str(path)], encoding="utf-8")
        stats["archive_listing"] = listing
        p = subprocess.Popen(
            [SEVEN, "x", "-so", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        source = p.stdout
    else:
        source = path
    ids = []
    hashes = []
    previous_timestamp = None
    numeric_by_type = {}
    pair_counts = collections.Counter()
    failure_pairs = collections.Counter()
    value_alarm = collections.Counter()
    print("START " + label, flush=True)
    for chunk_index, chunk in enumerate(
        pd.read_csv(
            source, dtype=str, keep_default_na=False, chunksize=500000, encoding="utf-8-sig"
        )
    ):
        if list(chunk.columns) != COLS:
            raise ValueError(f"Unexpected schema: {list(chunk.columns)}")
        n = len(chunk)
        stats["rows"] += n
        for col in COLS:
            stats["missing"][col] += int(chunk[col].eq("").sum())
        ch = chunk[COLS[1]]
        value = chunk[COLS[5]]
        alarm = chunk[COLS[4]]
        date = chunk[COLS[2]]
        tm = chunk[COLS[3]]
        fail = value.eq("Неисправен")
        is_alarm = alarm.isin(["t", "true"])
        types = ch.map(type_map).fillna("Нет в справочнике")
        systems = ch.map(system_map).fillna("Нет в справочнике")
        unknown = ~ch.isin(known)
        for key, series in [
            ("alarm", alarm),
            ("values", value),
            ("daily", date),
            ("channels", ch),
            ("types", types),
            ("systems", systems),
            ("daily_alarm", date[is_alarm]),
            ("daily_failure", date[fail]),
            ("failure_channels", ch[fail]),
            ("alarm_channels", ch[is_alarm]),
            ("failure_types", types[fail]),
            ("failure_systems", systems[fail]),
            ("failure_alarm", alarm[fail]),
        ]:
            stats[key].update(series.value_counts().to_dict())
        stats["unknown_rows"] += int(unknown.sum())
        numeric = pd.to_numeric(value, errors="coerce")
        numeric_valid = numeric.notna() & np.isfinite(numeric)
        stats["numeric_rows"] += int(numeric_valid.sum())
        stats["negative_numeric_rows"] += int(numeric.lt(0).sum())
        stats["text_alarm"].update(alarm[~numeric_valid].value_counts().to_dict())
        agg = (
            pd.DataFrame({"type": types[numeric_valid], "value": numeric[numeric_valid]})
            .groupby("type")["value"]
            .agg(["count", "min", "max", "sum"])
        )
        for k, row in agg.iterrows():
            old = numeric_by_type.setdefault(
                k, {"count": 0, "min": float("inf"), "max": float("-inf"), "sum": 0.0}
            )
            old["count"] += int(row["count"])
            old["min"] = min(old["min"], float(row["min"]))
            old["max"] = max(old["max"], float(row["max"]))
            old["sum"] += float(row["sum"])
        parsed_dates = pd.to_datetime(date, format="%Y-%m-%d", errors="coerce")
        stats["invalid_date_rows"] += int(parsed_dates.isna().sum())
        stats["invalid_time_rows"] += int(
            (~tm.str.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?")).sum()
        )
        timestamp = date + " " + tm
        stats["adjacent_time_reversals"] += int(timestamp.lt(timestamp.shift()).sum())
        if previous_timestamp is not None:
            stats["adjacent_time_reversals"] += int(timestamp.iloc[0] < previous_timestamp)
        previous_timestamp = timestamp.iloc[-1]
        rawids = pd.to_numeric(chunk[COLS[0]], errors="coerce")
        validids = rawids.notna() & rawids.mod(1).eq(0)
        stats["non_integer_ids"] += int((~validids).sum())
        ids.append(rawids[validids].to_numpy(dtype=np.int64))
        hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
        pair_counts.update(chunk.groupby([COLS[1], COLS[5]]).size().to_dict())
        failure_pairs.update(chunk.loc[fail].groupby([COLS[1], COLS[2]]).size().to_dict())
        value_alarm.update(chunk.groupby([COLS[5], COLS[4]]).size().to_dict())
        for key, frame in [
            ("examples", chunk),
            ("failure_examples", chunk.loc[fail]),
            ("unknown_examples", chunk.loc[unknown]),
        ]:
            if len(stats[key]) < 4:
                stats[key].extend(frame.head(4 - len(stats[key])).to_dict("records"))
        if chunk_index % 10 == 0:
            print(f"{label}: {stats['rows']:,} rows, {time.time() - start:.0f}s", flush=True)
    if archive:
        p.stdout.close()
        error = p.stderr.read().decode("utf-8", errors="replace")
        code = p.wait()
        stats["archive_exit_code"] = code
        if code != 0:
            raise RuntimeError(error)
    allids = np.concatenate(ids)
    del ids
    allids.sort()
    stats["duplicate_event_ids"] = int(np.sum(allids[1:] == allids[:-1]))
    stats["min_event_id"] = int(allids[0])
    stats["max_event_id"] = int(allids[-1])
    np.save(OUT / f"{label}.ids.npy", allids)
    del allids
    allhash = np.concatenate(hashes)
    del hashes
    allhash.sort()
    stats["duplicate_full_rows_hash64"] = int(np.sum(allhash[1:] == allhash[:-1]))
    del allhash
    stats["numeric_by_type"] = numeric_by_type
    stats["distinct_channels"] = len(stats["channels"])
    stats["unknown_channels"] = len(set(stats["channels"]) - known)
    stats["failure_channel_days"] = len(failure_pairs)
    stats["failure_channel_days_top"] = [
        [k[0], k[1], int(v)] for k, v in failure_pairs.most_common(15)
    ]
    stats["channel_value_top"] = [[k[0], k[1], int(v)] for k, v in pair_counts.most_common(30)]
    stats["value_alarm"] = [[k[0], k[1], int(v)] for k, v in value_alarm.most_common()]
    channel_values = collections.defaultdict(set)
    for channel, val in pair_counts:
        channel_values[channel].add(val)
    stats["one_value_channels"] = sum(len(v) == 1 for v in channel_values.values())
    stats["one_value_channels_100plus"] = sum(
        len(v) == 1 and stats["channels"][k] >= 100 for k, v in channel_values.items()
    )
    stats["elapsed_seconds"] = round(time.time() - start, 1)
    write_json(OUT / f"{label}.json", stats)
    print(f"DONE {label}: {stats['rows']:,} rows, {stats['elapsed_seconds']}s", flush=True)
    return label


def dictionaries():
    result = {}
    for path in sorted((ROOT / "data").glob("справочник*.csv")):
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        result[path.name] = {
            "rows": len(df),
            "columns": list(df.columns),
            "missing": {c: int(df[c].eq("").sum()) for c in df},
            "unique": {c: int(df[c].nunique()) for c in df},
            "duplicates": int(df.duplicated().sum()),
            "duplicate_keys": int(df.iloc[:, 0].duplicated().sum()),
            "frequencies": {c: df[c].value_counts().head(30).to_dict() for c in df.columns[1:]},
            "examples": df.head(5).to_dict("records"),
        }
        if "родитель" in df:
            result[path.name]["parents_missing_in_same_file"] = sorted(
                set(df["родитель"]) - set(df.iloc[:, 0])
            )
    write_json(OUT / "dictionaries.json", result)


if __name__ == "__main__":
    dictionaries()
    paths = [ROOT / "data" / "журнал_событий_пример.csv"] + sorted(
        (ROOT / "data").glob("*.7z")
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(profile, p) for p in paths]
        for future in concurrent.futures.as_completed(futures):
            future.result()
