"""Persisted A history and open episodes must preserve causal decisions."""

from copy import deepcopy
from datetime import datetime, timedelta
import unittest

from stage1.shadow.checkpoint import CheckpointError, parse_json, restore, snapshot
from stage1.shadow.stream import ShadowStream
from stage1.test_shadow_stream import T, baseline, feed, observation


class ShadowCheckpointTests(unittest.TestCase):
    def test_roundtrip_preserves_every_field_and_later_decisions(self):
        stream = ShadowStream(threshold=7.1)
        feed(stream, [*baseline(), observation(T)])
        stream.predict(T, ["c", "empty"])
        payload = snapshot(stream)
        resumed = restore(deepcopy(payload))
        self.assertEqual(snapshot(resumed), payload)
        for offset, text in ((1, "Неисправен"), (2, "Норма"), (25, "Неизвестно")):
            at = T + timedelta(hours=offset)
            group = [observation(at, text)]
            stream.observe_group(group)
            resumed.observe_group(group)
            self.assertEqual(
                stream.predict(at, ["c", "empty"]), resumed.predict(at, ["c", "empty"])
            )
        self.assertEqual(snapshot(stream), snapshot(resumed))

    def test_very_old_open_episode_survives_pruned_event_history(self):
        stream = ShadowStream(threshold=7.1)
        feed(
            stream,
            [
                observation(T - timedelta(days=100)),
                observation(T - timedelta(days=99), "Неисправен"),
                observation(T, text=None),
            ],
        )
        row = stream.predict(T, ["c"])[0]
        self.assertIn("registered_episode_active_at_t", row["admission_reasons"])
        self.assertEqual(len(stream.channels["c"].events), 1)
        resumed = restore(snapshot(stream))
        at = T + timedelta(hours=1)
        for value in (stream, resumed):
            value.observe_group([observation(at)])
        self.assertEqual(stream.predict(at, ["c"]), resumed.predict(at, ["c"]))
        self.assertEqual(resumed.channels["c"].completed, stream.channels["c"].completed)

    def test_checkpoint_requires_prediction_and_no_consumed_future_or_partial_group(self):
        stream = ShadowStream(threshold=7.1)
        feed(stream, baseline())
        with self.assertRaisesRegex(CheckpointError, "closed prediction hour"):
            snapshot(stream)
        stream.predict(T, ["c"])
        stream.observe_group([observation(T + timedelta(hours=1))])
        with self.assertRaisesRegex(CheckpointError, "closed prediction hour"):
            snapshot(stream)
        stream.predict(T + timedelta(hours=1), ["c"])
        stream.channels["c"].episodes._group.append(object())
        with self.assertRaisesRegex(CheckpointError, "partial channel-second"):
            snapshot(stream)

    def test_invalid_semantic_state_and_changed_policy_fail_closed(self):
        stream = ShadowStream(threshold=7.1)
        feed(stream, [*baseline(), observation(T)])
        stream.predict(T, ["c"])
        original = snapshot(stream)
        bad_values = []
        for name, value in (
            ("threshold", 6.1),
            ("accepted_rows", -1),
            ("pilot_version", "another-policy"),
            ("extra", True),
        ):
            payload = deepcopy(original)
            payload[name] = value
            bad_values.append(payload)
        for field, value in (
            ("past_count", 0),
            ("usable_count", True),
            ("first_usable_at", {"$datetime": "2026-01-01T00:00:00"}),
        ):
            payload = deepcopy(original)
            payload["channels"]["c"]["prefix"][field] = value
            bad_values.append(payload)
        for payload in bad_values:
            with self.subTest(payload=payload.keys()), self.assertRaises(CheckpointError):
                restore(payload)

    def test_late_and_duplicate_groups_remain_rejected_after_restore(self):
        stream = ShadowStream(threshold=7.1)
        feed(stream, [observation(T)])
        stream.predict(T, ["c"])
        stream = restore(snapshot(stream))
        with self.assertRaisesRegex(ValueError, "late event"):
            stream.observe_group([observation(T)])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            stream.predict(T, ["c"])

    def test_archive_boundary_still_discards_old_registered_state(self):
        stream = ShadowStream(threshold=7.1)
        at = datetime(2020, 12, 31, 23)
        feed(stream, [observation(at, "Неисправен")])
        stream.predict(at, ["c"])
        resumed = restore(snapshot(stream))
        resumed.observe_group([observation(datetime(2022, 1, 1))])
        row = resumed.predict(datetime(2022, 1, 1), ["c"])[0]
        self.assertNotIn("registered_episode_active_at_t", row["admission_reasons"])
        self.assertIn("insufficient_history", row["admission_reasons"])

    def test_duplicate_json_keys_nonfinite_numbers_and_unknown_tags_are_rejected(self):
        for encoded in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(CheckpointError):
                parse_json(encoded)
        with self.assertRaises(CheckpointError):
            restore({"$unknown": "not executable"})


if __name__ == "__main__":
    unittest.main()
