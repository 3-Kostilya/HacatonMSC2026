"""Produce reproducible real-trace review cases for the B2 episode catalog."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.build_registered_state_episodes import _validated_m1  # noqa: E402
from stage1.state_labeling.operational import source_is_full_archive  # noqa: E402


def _month_keys(low: datetime, high: datetime) -> list[tuple[int, int]]:
    year, month = low.year, low.month
    result = []
    while (year, month) <= (high.year, high.month):
        result.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return result


def _select(episodes: list[dict], count: int) -> list[dict]:
    # First pass covers types, years and onset statuses; remaining cases have
    # stable hash ordering so another machine selects the same examples.
    ordered = sorted(
        episodes,
        key=lambda row: hashlib.sha256(row["episode_id"].encode()).hexdigest(),
    )
    selected: list[dict] = []
    seen_types: set[str] = set()
    seen_years: set[int] = set()
    seen_status: set[str] = set()
    seen_end: set[str] = set()
    for row in ordered:
        if len(selected) >= count:
            break
        if (
            row["sensor_type"] not in seen_types
            or row["start_at"].year not in seen_years
            or row["onset_status"] not in seen_status
            or row["end_status"] not in seen_end
        ):
            selected.append(row)
            seen_types.add(row["sensor_type"])
            seen_years.add(row["start_at"].year)
            seen_status.add(row["onset_status"])
            seen_end.add(row["end_status"])
    chosen = {row["episode_id"] for row in selected}
    for row in ordered:
        if len(selected) >= count:
            break
        if row["episode_id"] not in chosen:
            selected.append(row)
            chosen.add(row["episode_id"])
    return selected


def _preview(rows: list[dict], center: datetime) -> dict:
    before = [row for row in rows if row["timestamp"] < center]
    equal = [row for row in rows if row["timestamp"] == center]
    after = [row for row in rows if row["timestamp"] > center]
    return {
        "before": before[-8:],
        "at": equal[:20],
        "after": after[:8],
        "total_before": len(before),
        "total_at": len(equal),
        "total_after": len(after),
    }


def audit(input_manifest: Path, episodes_path: Path, output: Path, count: int = 50) -> dict:
    files, _ = _validated_m1(input_manifest.resolve())
    by_month = {
        (int(path.parent.parent.name.removeprefix("year=")),
         int(path.parent.name.removeprefix("month="))): path
        for path in files
    }
    episodes = pq.read_table(episodes_path).to_pylist()
    selected = _select(episodes, count)
    if len(selected) != min(count, len(episodes)):
        raise ValueError("trace selection count differs from catalog")
    windows: dict[tuple[int, int], list[dict]] = defaultdict(list)
    case_info = {}
    for index, episode in enumerate(selected):
        case_id = f"episode-{index:02d}"
        case_info[case_id] = episode
        for kind, center in (("start", episode["start_at"]), ("end", episode["end_at"])):
            if center is None:
                continue
            low, high = center - timedelta(hours=24), center + timedelta(hours=24)
            for key in _month_keys(low, high):
                if key in by_month:
                    windows[key].append({
                        "case_id": case_id,
                        "kind": kind,
                        "channel_id": episode["channel_id"],
                        "center": center,
                        "low": low,
                        "high": high,
                    })
    observations: dict[tuple[str, str], list[dict]] = defaultdict(list)
    controls: dict[str, list[dict]] = {"normal": [], "unknown_type_fault": []}
    con = duckdb.connect(":memory:")
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=2")
    try:
        for path in files:
            if all(len(rows) >= 10 for rows in controls.values()):
                break
            for name, predicate in (
                ("normal", "value_state='Норма' AND sensor_type IS NOT NULL"),
                ("unknown_type_fault", "value_state='Неисправен' AND sensor_type IS NULL"),
            ):
                if len(controls[name]) >= 10:
                    continue
                rows = con.execute(
                    f"""SELECT row_id, channel_id, sensor_type, timestamp,
                               value_state, alarm, source
                        FROM read_parquet(?, hive_partitioning=false)
                        WHERE {predicate} ORDER BY timestamp, row_id LIMIT 20""",
                    [str(path)],
                ).to_arrow_table().to_pylist()
                controls[name].extend(
                    row for row in rows
                    if source_is_full_archive(row["source"], row["timestamp"])
                )
                controls[name] = controls[name][:10]
        for name, rows in controls.items():
            for index, row in enumerate(rows):
                center = row["timestamp"]
                low, high = center - timedelta(hours=24), center + timedelta(hours=24)
                for key in _month_keys(low, high):
                    if key in by_month:
                        windows[key].append({
                            "case_id": f"control-{name}-{index:02d}",
                            "kind": "control",
                            "channel_id": row["channel_id"],
                            "center": center,
                            "low": low,
                            "high": high,
                        })
        for key, items in sorted(windows.items()):
            con.register("review_windows", pa.Table.from_pylist(items))
            reader = con.execute(
                """
                SELECT w.case_id, w.kind, e.row_id, e.channel_id, e.sensor_type,
                       e.timestamp, e.value_state, e.alarm, e.source
                FROM read_parquet(?, hive_partitioning=false) e
                JOIN review_windows w
                  ON e.channel_id=w.channel_id
                 AND e.timestamp BETWEEN w.low AND w.high
                WHERE e.value_state IS NOT NULL
                ORDER BY w.case_id, w.kind, e.timestamp, e.row_id
                """,
                [str(by_month[key])],
            ).to_arrow_reader(batch_size=50_000)
            for batch in reader:
                for row in batch.to_pylist():
                    if source_is_full_archive(row["source"], row["timestamp"]):
                        observations[(row.pop("case_id"), row.pop("kind"))].append(row)
            con.unregister("review_windows")
            print(f"B2 trace review {key[0]}-{key[1]:02d}", flush=True)
    finally:
        con.close()

    cases = []
    failures = []
    for case_id, episode in case_info.items():
        start_rows = observations[(case_id, "start")]
        end_rows = observations[(case_id, "end")]
        start_found = any(
            row["row_id"] == episode["first_row_id"]
            and row["timestamp"] == episode["start_at"]
            and row["value_state"] == "Неисправен"
            for row in start_rows
        )
        end_found = episode["end_at"] is None or any(
            row["timestamp"] == episode["end_at"] and row["value_state"] == "Норма"
            for row in end_rows
        )
        if not (start_found and end_found):
            failures.append(case_id)
        cases.append({
            "case_id": case_id,
            "episode": episode,
            "start_trace": _preview(start_rows, episode["start_at"]),
            "end_trace": _preview(end_rows, episode["end_at"]) if episode["end_at"] else None,
            "checks": {"first_fault_found": start_found, "later_exact_norma_found": end_found},
        })
    control_traces = {}
    control_failures = []
    for name, rows in controls.items():
        control_traces[name] = [
            {
                "event": row,
                "trace": _preview(
                    observations[(f"control-{name}-{index:02d}", "control")],
                    row["timestamp"],
                ),
            }
            for index, row in enumerate(rows)
        ]
        if len(rows) < 10:
            control_failures.append(f"{name}:only_{len(rows)}")
        for index, item in enumerate(control_traces[name]):
            if not any(
                observed["row_id"] == item["event"]["row_id"]
                and observed["timestamp"] == item["event"]["timestamp"]
                for observed in observations[(f"control-{name}-{index:02d}", "control")]
            ):
                control_failures.append(f"{name}:{index}:event_missing")
    report = {
        "schema_version": "registered-state-b2-trace-review-v1",
        "selected_episode_count": len(selected),
        "catalog_episode_count": len(episodes),
        "selected_by_type": dict(Counter(row["sensor_type"] for row in selected)),
        "selected_by_year": dict(Counter(str(row["start_at"].year) for row in selected)),
        "selected_by_onset_status": dict(Counter(row["onset_status"] for row in selected)),
        "automated_trace_failures": failures,
        "automated_control_failures": control_failures,
        "controls": control_traces,
        "cases": cases,
        "review_limit": "Automated checks and trace packet; semantic review still requires inspection.",
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=lambda x: x.isoformat()) + "\n",
        encoding="utf-8",
    )
    if failures or control_failures:
        raise ValueError(f"trace checks failed: {failures + control_failures}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    report = audit(args.input_manifest, args.episodes, args.output, args.count)
    print(json.dumps({
        "selected_episode_count": report["selected_episode_count"],
        "automated_trace_failures": report["automated_trace_failures"],
        "controls": {key: len(rows) for key, rows in report["controls"].items()},
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
