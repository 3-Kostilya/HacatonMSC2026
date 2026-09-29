"""A-side checks for provenance-safe consumption of B2 episode artifacts."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_registered_state_episodes import EPISODE_SCHEMA
from analysis.r2_b2_handoff import _sha256, load_b2_for_a2
from stage1.state_labeling.registered_episodes import EPISODE_VERSION
from stage1.state_labeling.rules import RULESET_VERSION, TARGET_DEFINITION


T = datetime(2025, 6, 10)


def _episode(
    identifier: str,
    *,
    channel: str = "c",
    onset: str = "candidate_new_onset",
    end_status: str = "exact_norma",
    uncertain: bool = False,
) -> dict:
    return {
        "episode_id": identifier,
        "channel_id": channel,
        "sensor_type": "Датчик дыма",
        "target_kind": TARGET_DEFINITION["target_kind"],
        "start_at": T,
        "confirmed_at": T,
        "end_at": T + timedelta(hours=1) if end_status != "open_unknown" else None,
        "onset_status": onset,
        "end_status": end_status,
        "prior_normal_at": T - timedelta(hours=1),
        "first_row_id": int(identifier),
        "last_fault_at": T,
        "fault_message_count": 1,
        "uncertain_intervening_state": uncertain,
        "evidence": ["exact_neispraven"],
        "ruleset_version": RULESET_VERSION,
        "episode_version": EPISODE_VERSION,
    }


class B2HandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.m1 = self.root / "m1"
        self.m1.mkdir()
        self.catalog = self.root / "catalog"
        self.catalog.mkdir()
        manifest = {
            "schema_version": "ingestion-v1",
            "status": "complete",
            "scope": "full_supplied_sources",
            "input_rows": 4,
            "sources": [
                {
                    "path": "C:/data/ext-journal-2025.7z",
                    "sha256": "a" * 64,
                    "bytes": 4,
                    "rows_read": 4,
                }
            ],
            "dictionaries": [{"path": "C:/data/channels.csv", "sha256": "b" * 64}],
        }
        quality = {
            "dispositions": {"accepted": 4},
            "by_partition": [{"year": 2025, "month": 6, "rows": 4}],
            "join_status": {},
            "quality_flags": {},
        }
        (self.m1 / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (self.m1 / "data_quality.json").write_text(json.dumps(quality), encoding="utf-8")
        rows = [
            _episode("1"),
            _episode("2", onset="left_censored"),
            _episode("3", end_status="exact_norma_after_uncertainty", uncertain=True),
            _episode("4", end_status="open_unknown"),
            _episode("5", channel="other"),
        ]
        parquet = self.catalog / "registered_state_episodes.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=EPISODE_SCHEMA), parquet)
        report = self.catalog / "report.json"
        report.write_text(
            json.dumps(
                {
                    "episode_count": len(rows),
                    "ruleset_version": RULESET_VERSION,
                    "by_channel": {"c": {"episodes": 4}, "other": {"episodes": 1}},
                }
            ),
            encoding="utf-8",
        )
        self.catalog_manifest = self.catalog / "manifest.json"
        self._write_catalog_manifest()

    def _write_catalog_manifest(self, *, input_manifest: Path | None = None) -> None:
        input_manifest = input_manifest or self.m1 / "manifest.json"
        files = {}
        for name in ("registered_state_episodes.parquet", "report.json"):
            path = self.catalog / name
            files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        self.catalog_manifest.write_text(
            json.dumps(
                {
                    "schema_version": EPISODE_VERSION,
                    "status": "complete",
                    "ruleset_version": RULESET_VERSION,
                    "input_manifest_sha256": _sha256(input_manifest),
                    "input_data_quality_sha256": _sha256(self.m1 / "data_quality.json"),
                    "m1_accepted_rows": 4,
                    "episode_count": 5,
                    "files": files,
                }
            ),
            encoding="utf-8",
        )

    def test_only_unambiguous_completed_episodes_enter_a_features(self) -> None:
        selected = load_b2_for_a2(
            self.catalog, local_m1_manifest=self.m1 / "manifest.json", channels=["c"]
        )
        self.assertEqual(len(selected.episodes), 1)
        self.assertEqual(selected.episodes[0].channel_id, "c")
        self.assertEqual(selected.audit["selection_counts"]["selected_channel_episodes"], 4)
        self.assertEqual(selected.audit["selection_counts"]["unambiguous_completed"], 1)
        self.assertEqual(selected.audit["provenance_mode"], "identical_m1_manifest_and_quality")

    def test_cross_machine_m1_requires_matching_source_fingerprints(self) -> None:
        other = self.root / "b_manifest.json"
        contents = json.loads((self.m1 / "manifest.json").read_text(encoding="utf-8"))
        contents["sources"][0]["path"] = "D:/data/ext-journal-2025.7z"
        other.write_text(json.dumps(contents), encoding="utf-8")
        self._write_catalog_manifest(input_manifest=other)
        with self.assertRaisesRegex(ValueError, "provide B's M1"):
            load_b2_for_a2(
                self.catalog, local_m1_manifest=self.m1 / "manifest.json", channels=["c"]
            )
        selected = load_b2_for_a2(
            self.catalog,
            local_m1_manifest=self.m1 / "manifest.json",
            channels=["c"],
            b_m1_manifest=other,
            b_m1_quality=self.m1 / "data_quality.json",
        )
        self.assertEqual(
            selected.audit["provenance_mode"], "cross_machine_matching_sources_and_m1_quality"
        )
        contents["sources"][0]["sha256"] = "c" * 64
        other.write_text(json.dumps(contents), encoding="utf-8")
        self._write_catalog_manifest(input_manifest=other)
        with self.assertRaisesRegex(ValueError, "sources differ"):
            load_b2_for_a2(
                self.catalog,
                local_m1_manifest=self.m1 / "manifest.json",
                channels=["c"],
                b_m1_manifest=other,
                b_m1_quality=self.m1 / "data_quality.json",
            )

    def test_catalog_hash_mismatch_is_rejected(self) -> None:
        report = self.catalog / "report.json"
        report.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file size differs|file hash differs"):
            load_b2_for_a2(
                self.catalog, local_m1_manifest=self.m1 / "manifest.json", channels=["c"]
            )


if __name__ == "__main__":
    unittest.main()
