"""The sparse candidate relaxes statistics, not uncertainty or future guards."""

from datetime import datetime, timedelta
from itertools import groupby
import unittest

from ml.forecast.r6_rule import TERMS
from stage1.features.hourly import FeatureEvent
from stage1.features.sparse_admission import SparseAdmissionStream, candidate_from_past_snapshot
from stage1.shadow.stream import Observation, ShadowStream


T = datetime(2025, 12, 1)


def event(at, text="Норма", *, kind="Датчик дыма", numeric=None, flags=(), row_id=1):
    return Observation(
        row_id,
        FeatureEvent(
            "c",
            at,
            False,
            value_state=text,
            value_numeric=numeric,
            sensor_type=kind,
            quality_flags=flags,
        ),
        f"ext-journal-{at.year}.7z",
    )


def feed(stream, events):
    for _, group in groupby(
        sorted(events, key=lambda row: row.event.timestamp), key=lambda row: row.event.timestamp
    ):
        stream.observe_group(list(group))


def sparse_history(*, kind="Датчик дыма"):
    return [event(T - timedelta(days=10), kind=kind), event(T - timedelta(hours=30), kind=kind)]


class SparseAdmissionTests(unittest.TestCase):
    def test_sparse_history_can_be_eligible_without_baseline_or_recent_state(self):
        stream = SparseAdmissionStream()
        feed(stream, sparse_history())
        row = stream.evaluate(T, ["c"])[0]
        self.assertEqual(row["legacy_admission_status"], "unknown")
        self.assertEqual(row["admission_status"], "eligible")
        self.assertEqual(
            set(row["relaxed_data_reasons"]),
            {"baseline_unusable", "state_history_missing", "state_transitions_unavailable"},
        )
        self.assertEqual(row["availability_status"], "unknown")
        self.assertNotIn("rule_score", row)
        self.assertNotIn("warning_emitted", row)
        self.assertFalse(hasattr(stream, "predict"))

    def test_no_history_single_event_and_stale_normal_remain_unknown(self):
        for events, reason in (
            ([], "no_observations_in_current_archive_segment"),
            ([event(T)], "insufficient_history"),
            (
                [event(T - timedelta(days=10)), event(T - timedelta(days=8))],
                "no_recent_explicit_normal_at_t",
            ),
        ):
            stream = SparseAdmissionStream()
            feed(stream, events)
            row = stream.evaluate(T, ["c"])[0]
            self.assertEqual(row["admission_status"], "unknown")
            self.assertIn(reason, row["admission_reasons"])

    def test_fault_and_uncertain_text_are_not_recovered_by_silence_or_numbers(self):
        for text, status, reason in (
            ("Неисправен", "excluded", "registered_episode_active_at_t"),
            ("Непонятный текст", "unknown", "uncertain_past_registered_state"),
        ):
            stream = SparseAdmissionStream()
            feed(
                stream,
                [
                    *sparse_history(),
                    event(T - timedelta(hours=2), text),
                    event(T - timedelta(hours=1), None, numeric=0),
                ],
            )
            row = stream.evaluate(T, ["c"])[0]
            self.assertEqual(row["admission_status"], status)
            self.assertIn(reason, row["admission_reasons"])

    def test_unambiguous_normal_recovers_but_does_not_erase_quality_veto(self):
        stream = SparseAdmissionStream()
        feed(
            stream,
            [
                *sparse_history(),
                event(T - timedelta(hours=2), "Неисправен"),
                event(T - timedelta(hours=1)),
            ],
        )
        row = stream.evaluate(T, ["c"])[0]
        self.assertEqual(row["admission_status"], "eligible")
        self.assertEqual(row["completed_episode_count_168h"], 1)
        other = SparseAdmissionStream()
        feed(other, [*sparse_history(), event(T, flags=("channel_time_conflict",))])
        self.assertIn("quality_exclusions_24h", other.evaluate(T, ["c"])[0]["admission_reasons"])

    def test_same_second_conflict_is_order_independent(self):
        group = [event(T, row_id=2), event(T, "Неисправен", row_id=1)]
        outputs = []
        for order in (group, list(reversed(group))):
            stream = SparseAdmissionStream()
            feed(stream, sparse_history())
            stream.observe_group(order)
            outputs.append(stream.evaluate(T, ["c"])[0])
        self.assertEqual(*outputs)
        self.assertEqual(outputs[0]["admission_status"], "excluded")

    def test_unknown_or_changed_type_is_not_relaxed(self):
        for kind in (None, "ИБП"):
            stream = SparseAdmissionStream()
            feed(stream, [*sparse_history(), event(T, kind=kind)])
            self.assertEqual(stream.evaluate(T, ["c"])[0]["admission_status"], "unknown")

    def test_qa_guard_has_open_left_boundary_and_does_not_create_target(self):
        for kind, text, numeric in (
            ("Датчик температуры", None, -127),
            ("Газовый датчик", None, 101),
            ("Датчик дыма", "01.01.1970 03:00:01", None),
        ):
            stream = SparseAdmissionStream()
            feed(stream, [*sparse_history(kind=kind), event(T, text, kind=kind, numeric=numeric)])
            row = stream.evaluate(T, ["c"])[0]
            self.assertIn("qa_unusable_measurement_24h", row["admission_reasons"])
            self.assertEqual(row["registered_fault_text_count_24h"], 0)
            self.assertNotIn("target", row)
            later = stream.evaluate(T + timedelta(hours=24), ["c"])[0]
            self.assertEqual(later["blocking_qa_count_24h"], 0)
            if numeric is not None:
                self.assertEqual(later["admission_status"], "eligible")
            else:
                self.assertEqual(later["admission_status"], "unknown")
        for numeric in (-0.01, 1.0, 15.0):
            stream = SparseAdmissionStream()
            feed(
                stream,
                [
                    *sparse_history(kind="Газовый датчик"),
                    event(T, None, kind="Газовый датчик", numeric=numeric),
                ],
            )
            self.assertEqual(stream.evaluate(T, ["c"])[0]["blocking_qa_count_24h"], 0)

    def test_2021_gap_and_wrong_source_never_supply_history(self):
        stream = SparseAdmissionStream()
        feed(stream, [event(datetime(2020, 12, 31))])
        stream.evaluate(datetime(2021, 1, 1), ["c"])
        stream.observe_group([event(datetime(2022, 1, 1))])
        row = stream.evaluate(datetime(2022, 1, 1), ["c"])[0]
        self.assertIn("insufficient_history", row["admission_reasons"])
        other = SparseAdmissionStream()
        other.observe_group([Observation(1, event(T).event, "example.csv")])
        self.assertEqual(other.accepted_rows, 0)

    def test_prefix_and_future_tail_do_not_change_published_admission(self):
        published = []
        for tail in (
            [],
            [event(T + timedelta(hours=1), "Неисправен")],
            [event(T + timedelta(hours=1), "Неопределен", kind=None)],
        ):
            stream = SparseAdmissionStream()
            feed(stream, [*sparse_history(), event(T)])
            published.append(stream.evaluate(T, ["c"])[0])
            feed(stream, tail)
        self.assertEqual(published[0], published[1])
        self.assertEqual(published[0], published[2])

    def test_late_future_bad_watermarks_and_duplicate_channels_are_rejected(self):
        stream = SparseAdmissionStream()
        stream.evaluate(T, ["c"])
        with self.assertRaisesRegex(ValueError, "late event"):
            stream.observe_group([event(T)])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            stream.evaluate(T, ["c"])
        future = SparseAdmissionStream()
        feed(future, [event(T + timedelta(hours=1))])
        with self.assertRaisesRegex(ValueError, "future observations"):
            future.evaluate(T, ["c"])
        with self.assertRaisesRegex(ValueError, "whole hour"):
            SparseAdmissionStream().evaluate(T + timedelta(minutes=1), ["c"])
        with self.assertRaisesRegex(ValueError, "unique"):
            SparseAdmissionStream().evaluate(T, ["c", "c"])

    def test_future_labels_and_corrupt_snapshot_are_rejected(self):
        for field in ("target", "split", "horizon_end", "target_episode_id", "prior_normal_at"):
            with self.assertRaisesRegex(ValueError, "forbidden inputs"):
                SparseAdmissionStream().observe_records([{field: 1}])
        stream = SparseAdmissionStream()
        feed(stream, sparse_history())
        row = stream.evaluate(T, ["c"])[0]
        with self.assertRaisesRegex(ValueError, "future evidence"):
            candidate_from_past_snapshot({**row, "history_through": T + timedelta(hours=1)})
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            candidate_from_past_snapshot({**row, "admission_reasons": ["unexpected"]})

    def test_four_count_inputs_and_old_admission_match_unchanged_r6_engine(self):
        candidate, legacy = SparseAdmissionStream(), ShadowStream(threshold=7.1)
        events = [
            *sparse_history(),
            event(T - timedelta(hours=3)),
            event(T - timedelta(hours=2), "Неисправен"),
            event(T - timedelta(hours=1)),
        ]
        feed(candidate, events)
        feed(legacy, events)
        sparse, old = candidate.evaluate(T, ["c"])[0], legacy.predict(T, ["c"])[0]
        self.assertEqual({key: sparse[key] for key in TERMS}, {key: old[key] for key in TERMS})
        self.assertEqual(sparse["legacy_admission_status"], old["admission_status"])
        self.assertEqual(sparse["legacy_admission_reasons"], old["admission_reasons"])

    def test_scored_or_under_guarded_snapshot_cannot_be_reused_as_candidate(self):
        stream = SparseAdmissionStream()
        feed(stream, sparse_history())
        row = stream.evaluate(T, ["c"])[0]
        for field in ("target", "rule_score", "warning_emitted"):
            with self.assertRaisesRegex(ValueError, "unscored causal snapshot"):
                candidate_from_past_snapshot({**row, field: 1})
        for field in ("sensor_type", "last_explicit_normal_at", "history_through"):
            with self.assertRaisesRegex(ValueError, "lacks known type"):
                candidate_from_past_snapshot({**row, field: None})


if __name__ == "__main__":
    unittest.main()
