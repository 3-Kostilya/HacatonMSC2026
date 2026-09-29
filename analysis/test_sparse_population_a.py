"""Full SQL guards must match the accepted causal stream, including future tails."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from itertools import groupby
import random
import unittest

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_quality_improvement_a import counts_sql, create_qa_prefixes
from analysis.sparse_population_a import build_past_prefixes, decision_sql, finalize_sql
from analysis.sparse_population_a import retain_full_prefixes, slice_month_prefixes
from stage1.features.sparse_admission import SparseAdmissionStream


T = datetime(2025, 12, 1)


def row(at, text="Норма", *, kind="Датчик дыма", numeric=None, flags=(), channel="c", identity=1):
    return {
        "row_id": identity,
        "channel_id": channel,
        "timestamp": at,
        "alarm": False,
        "value_state": text,
        "value_numeric": numeric,
        "value_raw": text if text is not None else str(numeric),
        "sensor_type": kind,
        "quality_flags": list(flags),
        "source": f"ext-journal-{at.year}.7z",
    }


def calculate(records, times, *, sliced=False):
    with TemporaryDirectory() as temporary, duckdb.connect() as db:
        path = Path(temporary) / "raw.parquet"
        schema = pa.schema(
            [
                ("row_id", pa.int64()),
                ("channel_id", pa.string()),
                ("timestamp", pa.timestamp("us")),
                ("alarm", pa.bool_()),
                ("value_state", pa.string()),
                ("value_numeric", pa.float64()),
                ("value_raw", pa.string()),
                ("sensor_type", pa.string()),
                ("quality_flags", pa.list_(pa.string())),
                ("source", pa.string()),
            ]
        )
        pq.write_table(pa.Table.from_pylist(records, schema=schema), path)
        summary = build_past_prefixes(db, [str(path)])
        create_qa_prefixes(db, [path])
        if sliced:
            retain_full_prefixes(db)
            slice_month_prefixes(db, times[0].strftime("%Y-%m"))
        db.register(
            "keys",
            pa.Table.from_pylist(
                [
                    {
                        "channel_id": "c",
                        "prediction_time": t,
                        "archive_segment": 0 if t.year < 2021 else 1,
                    }
                    for t in times
                ]
            ),
        )
        db.execute("CREATE TEMP TABLE decisions_without_qa AS " + decision_sql())
        db.execute("CREATE TEMP TABLE qa_month AS " + counts_sql())
        result = (
            db.execute(finalize_sql() + " ORDER BY prediction_time").to_arrow_table().to_pylist()
        )
    return result, summary


def stream_rows(records, times):
    groups = iter(
        [
            list(group)
            for _, group in groupby(
                sorted(records, key=lambda r: (r["timestamp"], r["channel_id"])),
                lambda r: (r["timestamp"], r["channel_id"]),
            )
        ]
    )
    group = next(groups, None)
    stream = SparseAdmissionStream()
    result = []
    for at in times:
        while group is not None and group[0]["timestamp"] <= at:
            stream.observe_records(group)
            group = next(groups, None)
        result.append(stream.evaluate(at, ["c"])[0])
    return result


class FullSparsePopulationTests(unittest.TestCase):
    def assert_parity(self, records, times=None):
        times = times or [T, T + timedelta(hours=1), T + timedelta(hours=24)]
        actual, summary = calculate(records, times)
        expected = stream_rows(records, times)
        for a, e in zip(actual, expected, strict=True):
            for name in (
                "sensor_type",
                "admission_status",
                "admission_reasons",
                "last_explicit_normal_at",
                "blocking_qa_count_24h",
            ):
                self.assertEqual(a[name], e[name], (name, a, e))
            self.assertTrue(
                a["admission_evidence_through"] is None
                or a["admission_evidence_through"] <= a["prediction_time"]
            )
        return actual, summary

    def history(self, kind="Датчик дыма"):
        return [row(T - timedelta(days=10), kind=kind), row(T - timedelta(hours=30), kind=kind)]

    def test_sparse_active_uncertain_and_recovery(self):
        for text in ("Неисправен", "Неизвестное", "Обнаружен дым", "Норма"):
            self.assert_parity(
                [*self.history(), row(T - timedelta(hours=1), text), row(T + timedelta(hours=1))]
            )

    def test_conflicting_second_and_quality_exclusion(self):
        for records in (
            [*self.history(), row(T), row(T, "Неисправен", identity=2)],
            [*self.history(), row(T, flags=("channel_time_conflict",))],
            [
                row(T - timedelta(days=10), flags=("nonfinite_numeric",)),
                row(T, flags=("nonfinite_numeric",)),
            ],
        ):
            self.assert_parity(records)

    def test_qa_boundaries_and_numeric_metadata_uncertainty(self):
        for kind, text, numeric in (
            ("Датчик температуры", None, -127.0),
            ("Газовый датчик", None, 101.0),
            ("Датчик дыма", "01.01.1970 03:00:00", None),
            ("Газовый датчик", None, -0.01),
            ("Газовый датчик", None, 1.0),
        ):
            self.assert_parity([*self.history(kind), row(T, text, kind=kind, numeric=numeric)])
        self.assert_parity(
            [*self.history(), row(T, None, kind=None, numeric=1.0), row(T + timedelta(hours=1))]
        )
        self.assert_parity(
            [
                *self.history(),
                row(T),
                row(T, None, kind=None, numeric=1.0, identity=2),
                row(T + timedelta(hours=1)),
            ]
        )

    def test_type_change_uses_B_machine_and_future_type_does_not_change_prefix(self):
        source = [
            *self.history(),
            row(T, "Неисправен"),
            row(T + timedelta(hours=1), "Обнаружен дым", kind="ИБП"),
        ]
        result, summary = self.assert_parity(source)
        self.assertEqual(summary["type_changing_channel_segments"], 1)
        self.assertEqual(result[0]["admission_status"], "excluded")
        prefix, _ = calculate(source[:-1], [T])
        self.assertEqual(result[0], prefix[0])

    def test_future_tail_and_initial_cold_history(self):
        prefix = self.history()
        first, _ = calculate(prefix, [T])
        for tail in (
            row(T + timedelta(hours=1), "Неисправен"),
            row(T + timedelta(hours=1), kind=None),
        ):
            actual, _ = calculate([*prefix, tail], [T])
            self.assertEqual(first, actual)
        self.assert_parity([row(T)])
        self.assert_parity([row(T + timedelta(hours=1))])

    def test_archive_gap_resets_prefix_and_ignores_2021(self):
        at = datetime(2022, 1, 1)
        self.assert_parity(
            [row(datetime(2020, 12, 31)), row(datetime(2021, 12, 31)), row(at)],
            [at, at + timedelta(hours=1)],
        )

    def test_duplicate_timestamp_does_not_satisfy_two_unique_observations(self):
        self.assert_parity([row(T - timedelta(days=10)), row(T - timedelta(days=10), identity=2)])

    def test_month_slice_preserves_long_active_state_and_boundary_counts(self):
        records = [
            row(T - timedelta(days=200)),
            row(T - timedelta(days=180), "Неисправен"),
            row(T - timedelta(hours=24), flags=("channel_time_conflict",)),
            row(T + timedelta(hours=1)),
        ]
        times = [T, T + timedelta(hours=1), T + timedelta(hours=24)]
        full, _ = calculate(records, times)
        sliced, _ = calculate(records, times, sliced=True)
        self.assertEqual(full, sliced)

    def test_mixed_closed_groups_match_stream_at_every_hour(self):
        rng = random.Random(20260926)
        for case in range(12):
            records = self.history()
            texts = ["Норма", "Неисправен", "Обнаружен дым", "Неопределен", None]
            for hour in range(-4, 5):
                at = T + timedelta(hours=hour)
                for member in range(1 + int(rng.random() < 0.25)):
                    text = rng.choice(texts)
                    flags = ("channel_time_conflict",) if rng.random() < 0.1 else ()
                    records.append(row(at, text, numeric=1.0 if text is None else None,
                                       flags=flags, identity=100 * case + 2 * hour + member))
            self.assert_parity(records, [T + timedelta(hours=h) for h in range(5)])


if __name__ == "__main__":
    unittest.main()
