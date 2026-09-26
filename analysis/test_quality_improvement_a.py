"""Causal QA prefix parity, boundary/quality semantics and unchanged comparison cohorts."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_quality_improvement_a import (
    QA_NAMES,
    category_sql,
    counts_sql,
    coverage_month,
    create_qa_prefixes,
    expected_months,
    safe_path,
)
from analysis.verify_quality_improvement_a import validate_month
from stage1.features.hourly import FeatureEvent
from stage1.features.qa_values import QA_CATEGORIES, qa_window_counts
from stage1.value_quality import assess_value


T = datetime(2025, 2, 1)


def event(at, *, channel="c", raw="-127", numeric=-127.0, kind="Датчик температуры", flags=None):
    return {
        "row_id": 1,
        "channel_id": channel,
        "timestamp": at,
        "alarm": False,
        "value_numeric": numeric,
        "value_state": raw if numeric is None else None,
        "value_raw": raw,
        "sensor_type": kind,
        "quality_flags": flags or [],
        "source": f"ext-journal-{at.year}.7z",
    }


def calculate(source, points):
    with TemporaryDirectory() as temporary, duckdb.connect() as database:
        path = Path(temporary) / "observations.parquet"
        pq.write_table(pa.Table.from_pylist(source), path)
        create_qa_prefixes(database, [path])
        table = pa.Table.from_pylist(
            [
                {
                    "channel_id": channel,
                    "prediction_time": at,
                    "archive_segment": 0 if at.year < 2021 else 1,
                }
                for channel, at in points
            ]
        )
        database.register("keys", table)
        return database.execute(counts_sql()).to_arrow_table().to_pylist()


class QualityImprovementTests(unittest.TestCase):
    def export_fixture(self, root, *, corrupt=None):
        names = ["baseline_state_count", "state_count_24h", "state_transitions_24h", *QA_NAMES]
        qa = {"channel_id": "c", "prediction_time": T, **{name: 0 for name in QA_NAMES}}
        model = {
            "channel_id": "c",
            "prediction_time": T,
            "baseline_state_count": 1,
            "state_count_24h": 1,
            "state_transitions_24h": 0,
            **{name: qa[name] for name in QA_NAMES},
        }
        gate = {
            "channel_id": "c",
            "prediction_time": T,
            "qa_discrete_data_status": "eligible",
            "reason": None,
        }
        if corrupt == "qa_mismatch":
            model[QA_NAMES[-1]] = 1
        elif corrupt == "future_window":
            qa[QA_NAMES[0]] = 2
            model[QA_NAMES[0]] = 2
        elif corrupt == "key":
            gate["channel_id"] = "wrong"
        elif corrupt == "gate":
            gate["qa_discrete_data_status"] = "unknown"
        elif corrupt == "qa_type":
            qa[QA_NAMES[-1]] = 0.5
            model[QA_NAMES[-1]] = 0.5
        elif corrupt == "unknown_valid":
            model["baseline_state_count"] = 0
            gate["qa_discrete_data_status"] = "unknown"
            gate["reason"] = "qa_adjustment_removed_required_state_history"
        paths = [root / name for name in ("model.parquet", "qa.parquet", "gate.parquet")]
        for path, row in zip(paths, (model, qa, gate), strict=True):
            rows = [row, row] if corrupt == "duplicate_key" and path == paths[1] else [row]
            pq.write_table(pa.Table.from_pylist(rows), path)
        return paths, names

    def test_saved_exports_are_consistent(self):
        with TemporaryDirectory() as temporary, duckdb.connect() as database:
            paths, names = self.export_fixture(Path(temporary))
            self.assertEqual(validate_month(database, *paths, names), 1)

    def test_saved_unknown_gate_is_preserved(self):
        with TemporaryDirectory() as temporary, duckdb.connect() as database:
            paths, names = self.export_fixture(Path(temporary), corrupt="unknown_valid")
            self.assertEqual(validate_month(database, *paths, names), 1)

    def test_saved_export_checks_reject_corrupt_values_keys_gate_and_windows(self):
        for corruption in (
            "qa_mismatch",
            "future_window",
            "key",
            "gate",
            "qa_type",
            "duplicate_key",
        ):
            with self.subTest(corruption=corruption):
                with TemporaryDirectory() as temporary, duckdb.connect() as database:
                    paths, names = self.export_fixture(Path(temporary), corrupt=corruption)
                    with self.assertRaises(ValueError):
                        validate_month(database, *paths, names)

    def test_category_sql_matches_versioned_python_assessment(self):
        source = [
            event(T, numeric=value, kind=kind, raw=raw)
            for value, kind, raw in (
                (-127.0, "Датчик температуры", "-127"),
                (255.0, "Датчик температуры", "255"),
                (-127.0, "Датчик дыма", "-127"),
                (-0.01, "Газовый датчик", "-0.01"),
                (0.99, "Газовый датчик", "0.99"),
                (1.0, "Газовый датчик", "1"),
                (100.0, "Газовый датчик", "100"),
                (100.01, "Газовый датчик", "100.01"),
                (None, None, "\t01.01.1970 03:00:01\n"),
                (None, "Состояние охраны", "\u00a001.01.1970 03:00:00\u3000"),
                (None, "Датчик дыма", "Неисправен"),
            )
        ]
        with duckdb.connect() as database:
            database.register("source", pa.Table.from_pylist(source))
            actual = [
                row[0]
                for row in database.execute("SELECT " + category_sql() + " FROM source").fetchall()
            ]
        expected = [
            assess_value(row["sensor_type"], row["value_raw"], row["value_numeric"]).category
            for row in source
        ]
        self.assertEqual(actual, [value if value in QA_CATEGORIES else None for value in expected])

    def test_all_windows_match_python_with_closed_seconds_and_left_open_boundary(self):
        source = [event(T - timedelta(hours=hours)) for hours in (0, 1, 6, 24, 168, 169)]
        source += [
            event(T - timedelta(minutes=30)),
            event(T, flags=["channel_time_conflict"]),
            event(T, channel="other"),
            event(T + timedelta(seconds=1)),
        ]
        actual = calculate(source, [("c", T), ("empty", T)])
        by_channel = {row["channel_id"]: row for row in actual}
        expected = qa_window_counts(
            [
                FeatureEvent.from_clean_record(row, apply_qa_value_policy=True)
                for row in source
                if row["channel_id"] == "c"
            ],
            T,
        )
        self.assertEqual(
            {name: by_channel["c"][name] for name in QA_NAMES},
            {name: expected[name] for name in QA_NAMES},
        )
        self.assertTrue(all(by_channel["empty"][name] == 0 for name in QA_NAMES))

    def test_archive_gap_and_wrong_sources_never_enter_counts(self):
        at = datetime(2022, 1, 1)
        source = [event(datetime(2020, 12, 31)), event(datetime(2021, 12, 31)), event(at)]
        bad = event(at - timedelta(minutes=1))
        bad["source"] = "журнал_событий_пример.csv"
        source.append(bad)
        actual = calculate(source, [("c", at)])[0]
        self.assertEqual(actual["qa_temperature_service_code_candidate_count_168h"], 1)

    def test_future_tail_and_test_rows_cannot_change_previous_counts(self):
        prefix = [event(T - timedelta(hours=1)), event(T)]
        old = calculate(prefix, [("c", T)])
        new = calculate(
            [*prefix, event(T + timedelta(seconds=1)), event(datetime(2026, 1, 1))], [("c", T)]
        )
        self.assertEqual(old, new)

    def test_month_selection_excludes_test_and_2021_and_path_escape_is_rejected(self):
        months = expected_months()
        self.assertEqual(len(months), 72)
        self.assertFalse(any(month.startswith(("2021", "2026")) for month in months))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "escapes"):
                safe_path(root, "../outside.parquet")
            with self.assertRaisesRegex(ValueError, "permitted month"):
                safe_path(root, "year=2026/month=01/features.parquet", month="2025-01")

    def test_coverage_separates_past_data_status_from_conditional_label_cohort(self):
        with TemporaryDirectory() as temporary, duckdb.connect() as database:
            root = Path(temporary)
            features, statuses, keys = [root / name for name in ("features", "statuses", "keys")]
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"channel_id": channel, "prediction_time": T, "sensor_type": "Датчик дыма"}
                        for channel in ("admitted", "past_only", "unknown")
                    ]
                ),
                features,
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "channel_id": channel,
                            "prediction_time": T,
                            "discrete_data_status": "unknown"
                            if channel == "unknown"
                            else "eligible",
                            "numeric_data_status": "unknown",
                            "baseline_reasons": [],
                            "discrete_data_reasons": ["state_history_missing"]
                            if channel == "unknown"
                            else [],
                            "numeric_data_reasons": ["numeric_history_missing"],
                        }
                        for channel in ("admitted", "past_only", "unknown")
                    ]
                ),
                statuses,
            )
            pq.write_table(
                pa.Table.from_pylist([{"channel_id": "admitted", "prediction_time": T}]), keys
            )
            report = coverage_month(database, features, statuses, keys, 3)
            self.assertEqual(
                sum(group[-1] for group in report["groups"] if group[1] == "eligible"), 2
            )
            self.assertEqual(sum(group[-1] for group in report["groups"] if group[3]), 1)
            self.assertEqual(
                report["reasons"]["discrete_data_reasons"], {"state_history_missing": 1}
            )
            with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                coverage_month(database, features, statuses, keys, 4)


if __name__ == "__main__":
    unittest.main()
