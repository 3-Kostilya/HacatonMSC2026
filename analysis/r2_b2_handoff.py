"""Validate B2 provenance and select unambiguous past episodes for R2/A."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import ntpath
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from analysis.build_registered_state_episodes import EPISODE_SCHEMA
from stage1.features.r2 import CompletedEpisode
from stage1.state_labeling.operational import segment_at
from stage1.state_labeling.registered_episodes import EPISODE_VERSION
from stage1.state_labeling.rules import RULESET_VERSION, TARGET_DEFINITION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprints(rows: Iterable[dict[str, Any]]) -> dict[str, tuple[Any, ...]]:
    result = {}
    for row in rows:
        name = ntpath.basename(row["path"].replace("/", "\\"))
        if name in result:
            raise ValueError(f"duplicate M1 input basename: {name}")
        result[name] = row["sha256"], row.get("bytes"), row.get("rows_read")
    return result


def _verify_cross_machine_m1(
    *,
    local_manifest: Path,
    local_quality: Path,
    b_manifest: Path,
    b_quality: Path,
) -> None:
    local, other = _json(local_manifest), _json(b_manifest)
    local_q, other_q = _json(local_quality), _json(b_quality)
    if _fingerprints(local["sources"]) != _fingerprints(other["sources"]):
        raise ValueError("B2 M1 sources differ from local M1 sources")
    if _fingerprints(local["dictionaries"]) != _fingerprints(other["dictionaries"]):
        raise ValueError("B2 M1 dictionaries differ from local M1 dictionaries")
    for key in ("input_rows", "schema_version", "scope", "status"):
        if local.get(key) != other.get(key):
            raise ValueError(f"B2 M1 manifest differs on {key}")
    for key in ("dispositions", "by_partition", "join_status", "quality_flags"):
        if local_q.get(key) != other_q.get(key):
            raise ValueError(f"B2 M1 quality differs on {key}")


@dataclass(frozen=True, slots=True)
class CatalogSelection:
    episodes: tuple[CompletedEpisode, ...]
    audit: dict[str, Any]


def load_b2_for_a2(
    catalog_dir: Path,
    *,
    local_m1_manifest: Path,
    channels: Iterable[str],
    b_m1_manifest: Path | None = None,
    b_m1_quality: Path | None = None,
) -> CatalogSelection:
    """Accept only an intact full B2 catalog from equivalent M1 inputs.

    A duration from an uncertain or left-censored onset is only a lower bound,
    so the three completed-episode features use the conservative, unambiguous
    subset. Other episodes remain in B's catalog for target construction.
    """
    catalog_dir = catalog_dir.resolve()
    local_m1_manifest = local_m1_manifest.resolve()
    local_quality = local_m1_manifest.parent / "data_quality.json"
    catalog_manifest_path = catalog_dir / "manifest.json"
    catalog = _json(catalog_manifest_path)
    if (
        catalog.get("schema_version") != EPISODE_VERSION
        or catalog.get("status") != "complete"
        or catalog.get("ruleset_version") != RULESET_VERSION
    ):
        raise ValueError("B2 catalog version or publication status differs from R1/R2")
    same_m1 = catalog.get("input_manifest_sha256") == _sha256(local_m1_manifest)
    if same_m1:
        if catalog.get("input_data_quality_sha256") != _sha256(local_quality):
            raise ValueError("B2 and A have different M1 quality reports")
        provenance_mode = "identical_m1_manifest_and_quality"
    else:
        if b_m1_manifest is None or b_m1_quality is None:
            raise ValueError("B2 used another M1; provide B's M1 manifest and data_quality")
        b_m1_manifest, b_m1_quality = b_m1_manifest.resolve(), b_m1_quality.resolve()
        if catalog.get("input_manifest_sha256") != _sha256(b_m1_manifest) or catalog.get(
            "input_data_quality_sha256"
        ) != _sha256(b_m1_quality):
            raise ValueError("B2 catalog does not match the supplied B M1 audit files")
        _verify_cross_machine_m1(
            local_manifest=local_m1_manifest,
            local_quality=local_quality,
            b_manifest=b_m1_manifest,
            b_quality=b_m1_quality,
        )
        provenance_mode = "cross_machine_matching_sources_and_m1_quality"
    local_accepted = _json(local_quality)["dispositions"]["accepted"]
    if catalog.get("m1_accepted_rows") != local_accepted:
        raise ValueError("B2 catalog M1 accepted-row count differs from A")

    paths = {
        name: catalog_dir / name for name in ("registered_state_episodes.parquet", "report.json")
    }
    for name, path in paths.items():
        expected = catalog.get("files", {}).get(name)
        if expected is None or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"B2 catalog file size differs from manifest: {name}")
        if _sha256(path) != expected["sha256"]:
            raise ValueError(f"B2 catalog file hash differs from manifest: {name}")
    report = _json(paths["report.json"])
    parquet = pq.ParquetFile(paths["registered_state_episodes.parquet"])
    if not parquet.schema_arrow.equals(EPISODE_SCHEMA, check_metadata=False):
        raise ValueError("B2 episode Parquet schema differs from accepted version")
    if (
        parquet.metadata.num_rows != catalog.get("episode_count")
        or report.get("episode_count") != catalog.get("episode_count")
        or report.get("ruleset_version") != RULESET_VERSION
    ):
        raise ValueError("B2 episode counts or report ruleset disagree")

    selected_channels = sorted(set(channels))
    if not selected_channels:
        raise ValueError("R2/A needs at least one selected channel")
    table = pq.read_table(
        paths["registered_state_episodes.parquet"],
        columns=[
            "episode_id",
            "channel_id",
            "target_kind",
            "start_at",
            "end_at",
            "onset_status",
            "end_status",
            "uncertain_intervening_state",
            "ruleset_version",
            "episode_version",
        ],
        filters=[("channel_id", "in", selected_channels)],
    )
    by_channel_report = report.get("by_channel")
    if not isinstance(by_channel_report, dict):
        raise ValueError("B2 report lacks per-channel episode counts")
    expected_selected_rows = sum(
        by_channel_report.get(channel_id, {}).get("episodes", 0) for channel_id in selected_channels
    )
    if table.num_rows != expected_selected_rows:
        raise ValueError("B2 selected-channel Parquet rows differ from report")
    counters: Counter[str] = Counter()
    completed = []
    seen_ids = set()
    for row in table.to_pylist():
        if row["episode_id"] in seen_ids:
            raise ValueError("duplicate B2 episode ID in selected channels")
        seen_ids.add(row["episode_id"])
        if (
            row["channel_id"] not in selected_channels
            or row["ruleset_version"] != RULESET_VERSION
            or row["episode_version"] != EPISODE_VERSION
            or row["target_kind"] != TARGET_DEFINITION["target_kind"]
        ):
            raise ValueError("B2 selected episode has an unexpected channel or version")
        counters["selected_channel_episodes"] += 1
        end = row["end_at"]
        if end is None:
            counters["open_or_unresolved_end"] += 1
            continue
        if end <= row["start_at"] or segment_at(end) != segment_at(row["start_at"]):
            raise ValueError("B2 completed episode has invalid time boundaries")
        if row["onset_status"] != "candidate_new_onset":
            counters["excluded_uncertain_or_censored_onset"] += 1
            continue
        if row["end_status"] != "exact_norma" or row["uncertain_intervening_state"]:
            counters["excluded_uncertain_completion"] += 1
            continue
        completed.append(CompletedEpisode(row["channel_id"], row["start_at"], end))
        counters["unambiguous_completed"] += 1
    audit = {
        "catalog_manifest_sha256": _sha256(catalog_manifest_path),
        "catalog_input_manifest_sha256": catalog["input_manifest_sha256"],
        "provenance_mode": provenance_mode,
        "catalog_episode_count": catalog["episode_count"],
        "selected_channel_count": len(selected_channels),
        "selection_counts": dict(sorted(counters.items())),
        "feature_policy": "candidate_new_onset_and_exact_norma_without_uncertainty",
    }
    return CatalogSelection(tuple(completed), audit)
