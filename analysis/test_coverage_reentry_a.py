"""Research coverage must remain past-only, separate, conservative and auditable."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_coverage_reentry_a import evidence_sql, quality_prefix, summarize_episodes
from analysis.coverage_reentry_a import POLICIES, ResearchCoverageStream
from analysis.coverage_reentry_a import policy_sql, propose
from analysis.test_sparse_population_a import row
from analysis.verify_coverage_reentry_a import guard_violations_sql


T = datetime(2025, 12, 1)


def snapshot(reasons=("quality_exclusions_24h",), *, age_days=10, status="unknown", count=2):
    past = {
        "channel_id": "c",
        "prediction_time": T,
        "sensor_type": "Датчик дыма",
        "admission_status": status,
        "admission_reasons": list(reasons),
        "availability_status": "unknown",
        "last_explicit_normal_at": T - timedelta(hours=1),
        "admission_evidence_through": T,
        "first_usable_at": T - timedelta(days=age_days),
        "second_usable_at": T - timedelta(hours=2),
        "excluded_quality_count_24h": count,
        "ambiguous_seconds_24h": 0,
        "blocking_qa_count_24h": 0,
    }
    evidence = {
        "quality_rows_24h": count,
        "last_conflict_at": T - timedelta(hours=3),
        "last_hard_quality_at": None,
    }
    return past, evidence


def sql_rows(past, evidence):
    values = {**past, **evidence}
    schema = pa.schema(
        [
            (
                n,
                pa.timestamp("us")
                if n.endswith("_at") or n in {"prediction_time", "admission_evidence_through"}
                else pa.list_(pa.string())
                if n == "admission_reasons"
                else pa.int64()
                if n
                in {
                    "excluded_quality_count_24h",
                    "ambiguous_seconds_24h",
                    "blocking_qa_count_24h",
                    "quality_rows_24h",
                }
                else pa.string(),
            )
            for n in values
        ]
    )
    with duckdb.connect() as db:
        db.register("evidence", pa.Table.from_pylist([values], schema=schema))
        return db.execute(policy_sql()).to_arrow_table().to_pylist()[0]


class CoveragePolicyTests(unittest.TestCase):
    def parity(self, past, evidence):
        sql = sql_rows(past, evidence)
        for policy in POLICIES:
            expected = propose(past, evidence, policy)
            self.assertEqual(sql[policy + "_status"], expected["research_status"])
            self.assertEqual(sql[policy + "_reasons"], expected["research_reasons"])
        return sql

    def test_cold_start_requires_two_distinct_past_events(self):
        past, evidence = snapshot(("insufficient_history",), age_days=1, count=0)
        past["second_usable_at"] = T - timedelta(hours=2)
        self.assertEqual(self.parity(past, evidence)["cold_start_status"], "eligible")
        past["second_usable_at"] = None
        self.assertEqual(self.parity(past, evidence)["combined_status"], "unknown")

    def test_reentry_only_after_strictly_later_normal(self):
        past, evidence = snapshot()
        self.assertEqual(self.parity(past, evidence)["after_normal_status"], "eligible")
        for normal in (None, evidence["last_conflict_at"], T - timedelta(hours=4)):
            past["last_explicit_normal_at"] = normal
            self.assertEqual(self.parity(past, evidence)["combined_status"], "unknown")

    def test_simultaneous_fault_normal_and_row_order_do_not_reset(self):
        stream = ResearchCoverageStream()
        stream.observe_records([row(T - timedelta(days=10))])
        stream.observe_records([row(T - timedelta(hours=30))])
        stream.observe_records(
            [
                row(T, flags=("channel_time_conflict",)),
                row(T, "Неисправен", flags=("channel_time_conflict",), identity=2),
            ]
        )
        result = stream.evaluate(T, ["c"])[0]
        self.assertEqual(result["combined"]["research_status"], "excluded")
        self.assertIsNone(result["last_explicit_normal_at"])

    def test_hard_quality_and_severe_qa_never_cleared_by_normal(self):
        for field in ("last_hard_quality_at", "blocking_qa_count_24h", "ambiguous_seconds_24h"):
            past, evidence = snapshot()
            if field in evidence:
                evidence[field] = T - timedelta(hours=3)
            else:
                past[field] = 1
            self.assertEqual(self.parity(past, evidence)["combined_status"], "unknown")

    def test_hard_flag_exactly_24h_old_has_expired(self):
        past, evidence = snapshot()
        evidence["last_hard_quality_at"] = T - timedelta(hours=24)
        self.assertEqual(self.parity(past, evidence)["combined_status"], "eligible")

    def test_all_other_guards_remain_and_excluded_stays_excluded(self):
        for reason in (
            "registered_episode_active_at_t",
            "uncertain_past_registered_state",
            "unknown_type_observed_since_explicit_normal",
            "unknown_or_conflicting_sensor_type",
            "same_time_state_ambiguity",
            "no_recent_explicit_normal_at_t",
        ):
            past, evidence = snapshot(("quality_exclusions_24h", reason))
            self.assertEqual(self.parity(past, evidence)["combined_status"], "unknown")
        past, evidence = snapshot(status="excluded")
        self.assertEqual(self.parity(past, evidence)["combined_status"], "excluded")

    def test_ablations_do_not_silently_combine(self):
        past, evidence = snapshot(("insufficient_history", "quality_exclusions_24h"), age_days=1)
        actual = self.parity(past, evidence)
        self.assertEqual(actual["cold_start_status"], "unknown")
        self.assertEqual(actual["after_normal_status"], "unknown")
        self.assertEqual(actual["combined_status"], "eligible")

    def test_unknown_type_and_stale_normal_are_not_admitted(self):
        past, evidence = snapshot()
        past["sensor_type"] = None
        self.assertEqual(self.parity(past, evidence)["combined_status"], "unknown")
        past["sensor_type"] = "Датчик дыма"
        past["last_explicit_normal_at"] = T - timedelta(hours=169)
        self.assertEqual(self.parity(past, evidence)["combined_status"], "unknown")

    def test_future_labels_scores_and_extra_fields_are_rejected(self):
        past, evidence = snapshot()
        for name in ("target", "target_episode_id", "split", "rule_score", "label_available_at"):
            with self.assertRaisesRegex(ValueError, "only certified past"):
                propose({**past, name: 1}, evidence, "combined")

    def test_future_cross_gap_inconsistent_counts_and_timestamp_order_fail(self):
        past, evidence = snapshot()
        for changed in (
            {"second_usable_at": T + timedelta(seconds=1)},
            {"first_usable_at": datetime(2020, 12, 1)},
            {"second_usable_at": past["first_usable_at"]},
        ):
            with self.assertRaises(ValueError):
                propose({**past, **changed}, evidence, "combined")
        with self.assertRaisesRegex(ValueError, "raw quality count"):
            propose(past, {**evidence, "quality_rows_24h": 1}, "combined")

    def test_future_tail_does_not_change_earlier_admission(self):
        def replay(tail):
            stream = ResearchCoverageStream()
            for point in (T - timedelta(days=1), T - timedelta(hours=1)):
                stream.observe_records([row(point)])
            result = stream.evaluate(T, ["c"])[0]
            if tail:
                stream.observe_records([row(T + timedelta(hours=1), "Неисправен")])
            return result

        self.assertEqual(replay(False), replay(True))

    def test_gap_resets_history_and_quality(self):
        stream = ResearchCoverageStream()
        stream.observe_records([row(datetime(2020, 12, 1))])
        stream.observe_records([row(datetime(2022, 1, 1))])
        result = stream.evaluate(datetime(2022, 1, 1), ["c"])[0]
        self.assertEqual(result["first_usable_at"], datetime(2022, 1, 1))
        self.assertIsNone(result["second_usable_at"])
        self.assertEqual(result["combined"]["research_status"], "unknown")

    def test_unknown_and_no_history_inputs_not_promoted(self):
        stream = ResearchCoverageStream()
        result = stream.evaluate(T, ["none"])[0]
        self.assertEqual(result["combined"]["research_status"], "unknown")

    def test_raw_prefix_boundary_sources_and_future_flags(self):
        records = [
            row(T - timedelta(hours=24), flags=("nonfinite_numeric",)),
            row(T - timedelta(hours=3), flags=("channel_time_conflict",)),
            row(T - timedelta(hours=1)),
            row(T + timedelta(seconds=1), flags=("nonfinite_numeric",)),
        ]
        with TemporaryDirectory() as temporary, duckdb.connect() as db:
            path = Path(temporary) / "events.parquet"
            pq.write_table(pa.Table.from_pylist(records), path)
            quality_prefix(db, [str(path)])
            past, _ = snapshot(count=1)
            db.register("admission", pa.Table.from_pylist([{**past, "archive_segment": 1}]))
            got = db.execute(evidence_sql()).to_arrow_table().to_pylist()[0]
            self.assertEqual(got["quality_rows_24h"], 1)
            self.assertEqual(got["last_hard_quality_at"], T - timedelta(hours=24))
            self.assertEqual(got["last_conflict_at"], T - timedelta(hours=3))

    def test_episode_categories_are_disjoint_and_keep_full_denominator(self):
        rows = []
        for identity, statuses in enumerate(
            (
                ("eligible", "eligible", "eligible", "eligible"),
                ("unknown", "eligible", "unknown", "eligible"),
                ("unknown", "unknown", "eligible", "eligible"),
                ("unknown", "unknown", "unknown", "eligible"),
                ("unknown", "unknown", "unknown", "unknown"),
            )
        ):
            rows.append(
                {
                    "split": "validation",
                    "target_episode_id": str(identity),
                    "channel_id": "c",
                    "sensor_type": "Датчик дыма",
                    "admission_status": statuses[0],
                    "admission_reasons": []
                    if statuses[0] == "eligible"
                    else ["quality_exclusions_24h"],
                    **{p + "_status": s for p, s in zip(POLICIES, statuses[1:], strict=True)},
                    "combined_reasons": []
                    if statuses[-1] == "eligible"
                    else ["quality_exclusions_24h"],
                }
            )
        episodes, summary, _ = summarize_episodes(rows)
        self.assertEqual(summary["validation"]["all_episodes"], 5)
        self.assertEqual(summary["validation"]["combined_available"], 4)
        self.assertEqual(len({e["coverage_category"] for e in episodes}), 5)

    def test_independent_guard_checker_rejects_current_conflict_or_missing_second(self):
        cases = []
        for reasons, count in ((("insufficient_history",), 0), (("quality_exclusions_24h",), 2)):
            past, evidence = snapshot(reasons, age_days=1, count=count)
            cases.append(sql_rows(past, evidence))
        schema = pa.schema(
            [
                (
                    n,
                    pa.timestamp("us")
                    if n.endswith("_at") or n in {"prediction_time", "admission_evidence_through"}
                    else pa.list_(pa.string())
                    if n.endswith("reasons")
                    else pa.bool_()
                    if n in {"protected_ok", "can_cold_start", "can_reenter"}
                    else pa.int64()
                    if n.endswith("24h")
                    else pa.string(),
                )
                for n in cases[0]
            ]
        )
        with duckdb.connect() as db:
            db.register("delta", pa.Table.from_pylist(cases, schema=schema))
            self.assertEqual(db.execute(guard_violations_sql()).fetchone()[0], 0)
            bad = [dict(r) for r in cases]
            bad[0]["second_usable_at"] = None
            bad[1]["last_conflict_at"] = bad[1]["last_explicit_normal_at"]
            db.unregister("delta")
            db.register("delta", pa.Table.from_pylist(bad, schema=schema))
            self.assertEqual(db.execute(guard_violations_sql()).fetchone()[0], 2)


if __name__ == "__main__":
    unittest.main()
