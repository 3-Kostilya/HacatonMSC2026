"""Run the Day-3 baseline catalog over a normalized Parquet sample."""

from __future__ import annotations

import argparse
import collections
from datetime import datetime
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.contracts import NormalizedEvent  # noqa: E402
from stage1.detectors import RULESET_VERSION  # noqa: E402
from stage1.pipeline import evaluate_catalog  # noqa: E402
from stage1.registry import load_registry  # noqa: E402


DEFAULT_INPUT = ROOT / "output" / "stage1" / "normalized_sample.parquet"
DEFAULT_OUTPUT = ROOT / "output" / "stage1" / "baseline_catalog.json"


def _timestamp(value) -> datetime:
    return value.to_pydatetime() if hasattr(value, "to_pydatetime") else value


def read_events(path: Path) -> list[NormalizedEvent]:
    rows = pq.read_table(path).to_pylist()
    events = []
    for row in rows:
        if row["disposition"] != "accepted" or not row["channel_id"]:
            continue
        events.append(
            NormalizedEvent(
                channel_id=row["channel_id"],
                timestamp=_timestamp(row["timestamp"]),
                raw_value=row["raw_value"],
                alarm=row["alarm"],
                sensor_type=row["sensor_type"],
                source=row["source"],
                event_id=row["event_id"],
                object_id=row["object_id"],
                numeric_value=row["numeric_value"],
                quality_flags=tuple(row["quality_flags"] or ()),
            )
        )
    return events


def build_catalog(input_path: Path, output_path: Path) -> dict:
    registry = load_registry()
    events = read_events(input_path)
    outcomes = evaluate_catalog(events, registry=registry)
    counts = collections.Counter(outcome.detector_result.decision.value for outcome in outcomes)
    episode_counts = collections.Counter(
        episode.decision.value for outcome in outcomes for episode in outcome.detector_results
    )
    present_types = {outcome.sensor_type for outcome in outcomes}
    records = []
    for outcome in outcomes:
        audit = outcome.observability
        records.append(
            {
                "channel_id": outcome.channel_id,
                "sensor_type": outcome.sensor_type,
                "processing_mode": outcome.processing_mode,
                "detector": outcome.detector_result.to_record(),
                "detector_results": [episode.to_record() for episode in outcome.detector_results],
                "observability": {
                    "status": audit.status.value,
                    "reasons": list(audit.reasons),
                    "history_event_count": audit.history_event_count,
                    "history_sufficient": audit.history_sufficient,
                },
                "context_status": outcome.context_status,
                "context_reason": outcome.context_reason,
            }
        )
    result = {
        "ruleset": RULESET_VERSION,
        "input": str(input_path),
        "sample_only": True,
        "cadence_confirmed": False,
        "channels": len(outcomes),
        "types_present": len(present_types),
        "types_missing": sorted({policy.sensor_type for policy in registry} - present_types),
        "decisions": dict(counts),
        "episode_decisions": dict(episode_counts),
        "episodes": sum(episode_counts.values()),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_catalog(args.input, args.output)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
