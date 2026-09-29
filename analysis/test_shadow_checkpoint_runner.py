"""The real archive cursor and independent verifier, on tiny synthetic Parquet inputs."""

from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.replay_shadow_checkpoint import run
from analysis.replay_shadow_pilot import run as original_run
from analysis.train_r4_discrete_baselines import sha256
from analysis.verify_shadow_checkpoint import run as verify
from analysis.test_shadow_pilot import T, history, row
from stage1.shadow.checkpoint import CheckpointError


FREEZE = Path("ml/r6_frozen_rule_v1.json")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def fixture(root):
    m1 = root / "m1"
    m1.mkdir()
    write_json(m1 / "manifest.json", {"status": "complete"})
    path = m1 / "synthetic.parquet"
    source = [
        *history(),
        row(T),
        row(T + timedelta(minutes=30), text="Неисправен"),
        row(T + timedelta(hours=1, minutes=30)),
        row(T + timedelta(hours=2)),
    ]
    for item in source:
        item.update(object_id=None, join_status=None, quality_flags=[])
    pq.write_table(pa.Table.from_pylist(source), path)
    contract = json.loads(Path("ml/shadow_pilot_contract_v1.json").read_text(encoding="utf-8"))
    contract.update(
        source_m1_manifest_sha256=sha256(m1 / "manifest.json"),
        first_replay_end_exclusive=(T + timedelta(hours=3)).isoformat(),
    )
    contract_path = root / "contract.json"
    write_json(contract_path, contract)
    return m1, path, contract_path


class ShadowCheckpointRunnerTests(unittest.TestCase):
    def test_real_runner_resume_cursor_keeps_interstitial_events_and_matches_original_p1(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            m1, path, contract = fixture(root)
            args = {"m1_dir": m1, "freeze_path": FREEZE, "contract_path": contract}
            with patch(
                "analysis.replay_shadow_checkpoint._monthly_files", return_value=([path], [])
            ):
                run(**args, output_dir=root / "continuous")
                run(**args, output_dir=root / "prefix", stop_at=T + timedelta(hours=1))
                run(**args, output_dir=root / "resumed", checkpoint_in=root / "prefix")
            with patch("analysis.replay_shadow_pilot._monthly_files", return_value=([path], [])):
                original_run(**args, output_dir=root / "original")
            report = verify(
                continuous_dir=root / "continuous",
                resumed_dir=root / "resumed",
                reference_dir=root / "original",
                freeze_path=FREEZE,
                output_dir=root / "check",
            )
            self.assertEqual(report["rows_checked"], 3)
            self.assertEqual(report["final_source_cursor"]["accepted_rows"], 30)
            self.assertEqual(report["accepted_p1_reference"]["decision_mismatches"], 0)
            self.assertTrue(all(report["deterministic_files_bitwise_equal"].values()))

    def test_mutated_source_and_mutated_contract_are_not_resumed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            m1, path, contract = fixture(root)
            args = {"m1_dir": m1, "freeze_path": FREEZE, "contract_path": contract}
            with patch(
                "analysis.replay_shadow_checkpoint._monthly_files", return_value=([path], [])
            ):
                run(**args, output_dir=root / "prefix", stop_at=T + timedelta(hours=1))
                original = path.read_bytes()
                path.write_bytes(original + b"source changed")
                with self.assertRaisesRegex(CheckpointError, "lineage"):
                    run(**args, output_dir=root / "bad-source", checkpoint_in=root / "prefix")
                path.write_bytes(original)
                altered = json.loads(contract.read_text(encoding="utf-8"))
                altered["status"] = "changed-contract-version"
                write_json(contract, altered)
                with self.assertRaisesRegex(CheckpointError, "lineage"):
                    run(**args, output_dir=root / "bad-contract", checkpoint_in=root / "prefix")

    def test_invalid_stop_hour_and_changed_admission_settings_are_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            m1, path, contract = fixture(root)
            args = {"m1_dir": m1, "freeze_path": FREEZE, "contract_path": contract}
            with patch(
                "analysis.replay_shadow_checkpoint._monthly_files", return_value=([path], [])
            ):
                with self.assertRaisesRegex(CheckpointError, "whole hour"):
                    run(**args, output_dir=root / "bad-hour", stop_at=T + timedelta(minutes=30))
                altered = deepcopy(json.loads(contract.read_text(encoding="utf-8")))
                altered["event_retention_days"] = 1
                write_json(contract, altered)
                with self.assertRaisesRegex(CheckpointError, "policy differs"):
                    run(**args, output_dir=root / "bad-settings")


if __name__ == "__main__":
    unittest.main()
