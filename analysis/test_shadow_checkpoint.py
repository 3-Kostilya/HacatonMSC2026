"""Atomic A/B restart, output prefix and positive-warning cooldown integration."""

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from analysis.replay_shadow_pilot import observation_groups
from analysis.shadow_checkpoint_bundle import ReplaySession, load_bundle, save_bundle
from ml.forecast.shadow_pilot import ShadowPolicy
from stage1.shadow.checkpoint import CheckpointError, canonical, parse_json
from analysis.train_r4_discrete_baselines import sha256
from analysis.test_shadow_pilot import T, history, row


SOURCE = {"mode": "immutable_test_archive", "manifest": "test-only"}


def session(*, start=T, end=T + timedelta(hours=27)):
    return ReplaySession(
        source_identity=deepcopy(SOURCE),
        channels=["c", "empty"],
        start=start,
        end=end,
        policy=ShadowPolicy.from_freeze(Path("ml/r6_frozen_rule_v1.json")),
    )


def events():
    return sorted(
        [
            *history(),
            row(T - timedelta(hours=3)),
            *[row(T - timedelta(hours=2), text="Неисправен", row_id=100 + i) for i in range(4)],
            row(T - timedelta(hours=1)),
            *[row(T + timedelta(hours=22), text="Неисправен", row_id=200 + i) for i in range(4)],
            row(T + timedelta(hours=23)),
        ],
        key=lambda item: (item["timestamp"], item["channel_id"]),
    )


def advance(value, source, stop_at):
    # Resume source filtering deliberately mirrors the real SQL cursor, not an iterator offset.
    remaining = [
        item
        for item in source
        if value.a.watermark is None or item["timestamp"] > value.a.watermark
    ]
    groups = iter(observation_groups(remaining))
    group = next(groups, None)
    while value.next_prediction < stop_at:
        while group is not None and group[0].event.timestamp <= value.next_prediction:
            value.observe_group(group)
            group = next(groups, None)
        value.predict_hour()


class ShadowCheckpointIntegrationTests(unittest.TestCase):
    def test_failed_publication_keeps_previous_commit_and_never_restores_partial_files(self):
        value = session()
        advance(value, events(), T + timedelta(hours=1))
        with TemporaryDirectory() as temporary:
            previous, failed = Path(temporary) / "previous", Path(temporary) / "failed"
            save_bundle(value, previous)
            advance(value, events(), T + timedelta(hours=2))
            with patch.object(Path, "rename", side_effect=OSError("simulated interruption")):
                with self.assertRaises(OSError):
                    save_bundle(value, failed)
            self.assertFalse(failed.exists())
            with self.assertRaisesRegex(CheckpointError, "unpublished"):
                load_bundle(
                    failed.with_name("failed.inprogress"),
                    source_identity=SOURCE,
                    policy=value.policy,
                )
            restored = load_bundle(previous, source_identity=SOURCE, policy=value.policy)
            self.assertEqual(restored.next_prediction, T + timedelta(hours=1))

    def test_positive_warnings_cooldown_unknowns_and_final_state_equal_after_restart(self):
        full, split = session(), session()
        advance(full, events(), full.end)
        advance(split, events(), T + timedelta(hours=1))
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "closed-hour"
            save_bundle(split, directory)
            resumed = load_bundle(directory, source_identity=SOURCE, policy=split.policy)
            advance(resumed, events(), resumed.end)
            self.assertEqual(resumed.predictions, full.predictions)
            self.assertEqual(resumed.decisions, full.decisions)
            self.assertEqual(resumed.checkpoint(), full.checkpoint())
            warnings = [item for item in resumed.decisions if item["shadow_warning"]]
            self.assertEqual(
                [item["prediction_time"] for item in warnings],
                [T.isoformat(), (T + timedelta(hours=24)).isoformat()],
            )
            self.assertTrue(
                any(item["warning_reason"] == "channel_cooldown" for item in resumed.decisions)
            )

    def test_restart_mid_active_episode_across_month_and_day_baseline_boundary(self):
        start = datetime(2025, 11, 30, 23)
        source = sorted(
            [
                *history(),
                row(start - timedelta(hours=1)),
                row(start, text="Неисправен"),
                row(start + timedelta(hours=1)),
            ],
            key=lambda item: item["timestamp"],
        )
        full = session(start=start, end=start + timedelta(hours=3))
        split = session(start=start, end=full.end)
        advance(full, source, full.end)
        advance(split, source, start + timedelta(hours=1))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "month-boundary"
            save_bundle(split, path)
            resumed = load_bundle(path, source_identity=SOURCE, policy=split.policy)
            advance(resumed, source, resumed.end)
            self.assertEqual(resumed.predictions, full.predictions)
            self.assertEqual(resumed.checkpoint(), full.checkpoint())

    def test_corruption_missing_file_unpublished_bundle_and_source_change_stop_restore(self):
        value = session()
        advance(value, events(), T + timedelta(hours=1))
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "checkpoint"
            save_bundle(value, directory)
            with self.assertRaisesRegex(CheckpointError, "lineage"):
                load_bundle(directory, source_identity={"manifest": "changed"}, policy=value.policy)
            file = directory / "checkpoint.json"
            file.write_bytes(file.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(CheckpointError, "integrity"):
                load_bundle(directory, source_identity=SOURCE, policy=value.policy)
            pending = Path(temporary) / "checkpoint.inprogress"
            pending.mkdir()
            with self.assertRaisesRegex(CheckpointError, "unpublished"):
                load_bundle(pending, source_identity=SOURCE, policy=value.policy)
            missing = Path(temporary) / "missing"
            with self.assertRaisesRegex(CheckpointError, "unavailable"):
                load_bundle(missing, source_identity=SOURCE, policy=value.policy)

    def test_manifest_rehashed_but_inconsistent_cursor_still_fails(self):
        value = session()
        advance(value, events(), T + timedelta(hours=1))
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "checkpoint"
            save_bundle(value, directory)
            path = directory / "checkpoint.json"
            payload = parse_json(path.read_bytes())
            payload["source_cursor"]["closed_through"] = "2026-01-01T00:00:00"
            path.write_bytes(canonical(payload))
            manifest_path = directory / "manifest.json"
            manifest = parse_json(manifest_path.read_bytes())
            manifest["files_sha256"][path.name] = sha256(path)
            manifest_path.write_bytes(canonical(manifest))
            with self.assertRaisesRegex(CheckpointError, "cursor"):
                load_bundle(directory, source_identity=SOURCE, policy=value.policy)

    def test_in_memory_uncommitted_progress_is_replayed_from_previous_commit(self):
        full, value = session(), session()
        advance(full, events(), full.end)
        advance(value, events(), T + timedelta(hours=1))
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "committed"
            save_bundle(value, directory)
            # Simulate a crash after more decisions, before their bundle publication.
            advance(value, events(), T + timedelta(hours=25))
            resumed = load_bundle(directory, source_identity=SOURCE, policy=value.policy)
            advance(resumed, events(), resumed.end)
            self.assertEqual(resumed.decisions, full.decisions)
            self.assertEqual(resumed.checkpoint(), full.checkpoint())
            with self.assertRaises(FileExistsError):
                save_bundle(resumed, directory)

    def test_future_and_late_source_groups_are_rejected_after_restore(self):
        value = session()
        advance(value, events(), T + timedelta(hours=1))
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "checkpoint"
            save_bundle(value, directory)
            resumed = load_bundle(directory, source_identity=SOURCE, policy=value.policy)
            with self.assertRaisesRegex(ValueError, "late event"):
                resumed.observe_group(next(observation_groups([row(T)])))
            resumed.observe_group(next(observation_groups([row(T + timedelta(hours=2))])))
            with self.assertRaisesRegex(ValueError, "future observations"):
                resumed.predict_hour()


if __name__ == "__main__":
    unittest.main()
