"""Independently check R2/A episode features against B2 and real trace cases."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import sys
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.r2_b2_handoff import _sha256, load_b2_for_a2  # noqa: E402
from stage1.features.r2 import (  # noqa: E402
    CompletedEpisode,
    R2_VERSION,
    build_state_history_rows,
    validate_r2_table,
)
from stage1.state_labeling.operational import segment_at  # noqa: E402
from stage1.state_labeling.rules import RULESET_VERSION  # noqa: E402


EPISODE_FIELDS = (
    "last_completed_episode_end_age_seconds",
    "completed_episode_count_168h",
    "completed_episode_mean_duration_seconds_168h",
    "episode_history_status",
)


def _expected(
    episodes: list[CompletedEpisode], channel_id: str, at: datetime
) -> tuple[float | None, int, float | None]:
    segment = segment_at(at)
    past = [
        item
        for item in episodes
        if item.channel_id == channel_id
        and segment_at(item.end_at) == segment
        and item.end_at <= at
    ]
    last_age = (at - max(item.end_at for item in past)).total_seconds() if past else None
    recent = [item for item in past if item.end_at > at - timedelta(hours=168)]
    mean_duration = (
        sum((item.end_at - item.start_at).total_seconds() for item in recent) / len(recent)
        if recent
        else None
    )
    return last_age, len(recent), mean_duration


def _same_number(actual: float | None, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-6)


def _matches(row: dict[str, Any], expected: tuple[float | None, int, float | None]) -> bool:
    age, count, mean = expected
    return (
        _same_number(row["last_completed_episode_end_age_seconds"], age)
        and row["completed_episode_count_168h"] == count
        and _same_number(row["completed_episode_mean_duration_seconds_168h"], mean)
        and row["episode_history_status"] == "unambiguous_completed_only"
    )


def _probe_row(channel_id: str, at: datetime) -> dict[str, Any]:
    return {
        "run_id": "r2-b2-asof-audit",
        "channel_id": channel_id,
        "prediction_time": at,
        "sensor_type": "Датчик дыма",
        "availability_status": "unknown",
        "availability_reasons": ["cadence_unknown"],
        "baseline_status": "unknown",
        "baseline_state_count": 0,
        "baseline_numeric_count": 0,
        "baseline_numeric_median": None,
        "baseline_numeric_mad": None,
        "state_count_24h": 0,
        "numeric_count_24h": 0,
        "numeric_median_24h": None,
        "state_transitions_24h": None,
        "excluded_quality_count_24h": 0,
        "window_reasons_24h": [],
    }


def _r2_table(directory: Path, *, m1_sha: str) -> tuple[Any, dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != R2_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("ruleset_version") != RULESET_VERSION
        or manifest.get("source_m1_manifest_sha256") != m1_sha
    ):
        raise ValueError("R2 artifact version, status or M1 provenance differs")
    for name in ("state_history.parquet", "report.json"):
        path = directory / name
        expected = manifest.get("files", {}).get(name)
        if (
            expected is None
            or path.stat().st_size != expected["bytes"]
            or _sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"R2 artifact file differs from manifest: {name}")
    table = pq.read_table(directory / "state_history.parquet")
    validate_r2_table(table)
    if table.num_rows != manifest.get("row_count"):
        raise ValueError("R2 physical row count differs from manifest")
    return table, manifest


def audit(
    *,
    a2_dir: Path,
    baseline_dir: Path,
    integrated_dir: Path,
    b2_dir: Path,
    m1_manifest: Path,
    rechecked_traces: Path,
    output: Path,
) -> dict[str, Any]:
    a2_manifest = json.loads((a2_dir / "manifest.json").read_text(encoding="utf-8"))
    channels = a2_manifest["config"]["channels"]
    m1_sha = _sha256(m1_manifest)
    baseline, _ = _r2_table(baseline_dir, m1_sha=m1_sha)
    integrated, integrated_manifest = _r2_table(integrated_dir, m1_sha=m1_sha)
    if integrated_manifest.get("episode_catalog", {}).get("catalog_manifest_sha256") != _sha256(
        b2_dir / "manifest.json"
    ):
        raise ValueError("R2 integrated artifact was built from another B2 catalog")
    if baseline.num_rows != integrated.num_rows or not baseline.drop(EPISODE_FIELDS).equals(
        integrated.drop(EPISODE_FIELDS)
    ):
        raise ValueError("R2 integration changed non-episode features or row order")
    if integrated.num_rows != a2_manifest["feature_rows"]:
        raise ValueError("R2 integrated row count differs from A2")
    catalog = load_b2_for_a2(b2_dir, local_m1_manifest=m1_manifest, channels=channels)
    selected_episodes = list(catalog.episodes)
    failures = []
    for row in integrated.to_pylist():
        if not _matches(
            row, _expected(selected_episodes, row["channel_id"], row["prediction_time"])
        ):
            failures.append(f"hour:{row['channel_id']}:{row['prediction_time'].isoformat()}")

    trace_file = b2_dir / "trace_review.json"
    b_trace = json.loads(trace_file.read_text(encoding="utf-8"))
    a_trace = json.loads(rechecked_traces.read_text(encoding="utf-8"))
    if b_trace != a_trace:
        raise ValueError("B2 trace packet differs from independent M1 recheck")
    if (
        len(a_trace["cases"]) != 50
        or any(len(a_trace["controls"][name]) != 10 for name in ("normal", "unknown_type_fault"))
        or a_trace["automated_trace_failures"]
        or a_trace["automated_control_failures"]
    ):
        raise ValueError("B2 trace packet lacks required cases or controls")
    trace_channels = {case["episode"]["channel_id"] for case in a_trace["cases"]}
    trace_catalog = load_b2_for_a2(b2_dir, local_m1_manifest=m1_manifest, channels=trace_channels)
    by_channel: dict[str, list[CompletedEpisode]] = defaultdict(list)
    for episode in trace_catalog.episodes:
        by_channel[episode.channel_id].append(episode)
    asof_checks = 0
    unambiguous_cases = 0
    for case in a_trace["cases"]:
        episode = case["episode"]
        channel_id = episode["channel_id"]
        start = datetime.fromisoformat(episode["start_at"])
        end = datetime.fromisoformat(episode["end_at"]) if episode["end_at"] else None
        if not all(case["checks"].values()):
            failures.append(f"trace:{case['case_id']}")
        accepted = (
            end is not None
            and episode["onset_status"] == "candidate_new_onset"
            and episode["end_status"] == "exact_norma"
            and not episode["uncertain_intervening_state"]
        )
        if accepted:
            unambiguous_cases += 1
            if not any(
                item.channel_id == channel_id and item.start_at == start and item.end_at == end
                for item in by_channel[channel_id]
            ):
                failures.append(f"missing_unambiguous_case:{case['case_id']}")
        moments = [start]
        if end is not None:
            moments.extend((end - timedelta(microseconds=1), end))
        for at in moments:
            row = build_state_history_rows(
                [_probe_row(channel_id, at)],
                [],
                source_a2_manifest_sha256="0" * 64,
                completed_episodes=by_channel[channel_id],
            ).to_pylist()[0]
            if not _matches(row, _expected(by_channel[channel_id], channel_id, at)):
                failures.append(f"asof:{case['case_id']}:{at.isoformat()}")
            asof_checks += 1

    result = {
        "status": "pass" if not failures else "fail",
        "source_a2_manifest_sha256": _sha256(a2_dir / "manifest.json"),
        "source_b2_manifest_sha256": _sha256(b2_dir / "manifest.json"),
        "source_b2_trace_sha256": _sha256(trace_file),
        "source_integrated_r2_manifest_sha256": _sha256(integrated_dir / "manifest.json"),
        "hourly_rows_checked": integrated.num_rows,
        "non_episode_fields_unchanged": True,
        "trace_cases_checked": len(a_trace["cases"]),
        "trace_unambiguous_completed_cases": unambiguous_cases,
        "normal_controls": len(a_trace["controls"]["normal"]),
        "unknown_type_controls": len(a_trace["controls"]["unknown_type_fault"]),
        "trace_asof_probes": asof_checks,
        "failed_checks": failures[:50],
        "limitations": [
            "Twenty selected channels are a QA slice, not a train/test population.",
            "Completed-episode features intentionally exclude censored and ambiguous episodes.",
            "Trace checks verify registered messages and causality, not physical failures.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("R2 A/B2 audit output already exists")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise ValueError(f"R2 A/B2 as-of audit failed: {len(failures)} checks")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a2-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--integrated-dir", required=True, type=Path)
    parser.add_argument("--b2-dir", required=True, type=Path)
    parser.add_argument("--m1-manifest", required=True, type=Path)
    parser.add_argument("--rechecked-traces", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
