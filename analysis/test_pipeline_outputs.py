from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.prepare_stage1_sample import SCHEMA
from analysis.run_stage1_baseline import build_catalog
from analysis.run_synthetic_benchmark import VARIANTS, load_inputs, run_variant
from stage1.contracts import NormalizedEvent
from stage1.pipeline import evaluate_channel
from stage1.simulation import build_suite, write_suite


BASE = datetime(2026, 1, 1)


def numeric_event(index: int, value: float) -> NormalizedEvent:
    return NormalizedEvent(
        channel_id="review",
        timestamp=BASE + timedelta(minutes=index),
        raw_value=str(value),
        numeric_value=value,
        alarm=False,
        sensor_type="Датчик температуры",
        source="fixture.csv",
    )


class PipelineOutputTests(unittest.TestCase):
    def test_catalog_serializes_every_pipeline_episode(self):
        events = [
            numeric_event(index, value)
            for index, value in enumerate([10] * 12 + [20] * 3 + [10] * 20 + [20] * 3 + [10] * 20)
        ]
        expected = evaluate_channel(events).detector_results
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, event in enumerate(events):
                row = event.to_record()
                row.update(
                    timestamp=event.timestamp,
                    disposition="accepted",
                    source_row=index + 2,
                    duplicate_of_source=None,
                    duplicate_of_source_row=None,
                )
                rows.append(row)
            source = root / "events.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), source)
            catalog = build_catalog(source, root / "catalog.json")

        serialized = catalog["records"][0]["detector_results"]
        self.assertEqual(catalog["episodes"], len(expected))
        self.assertEqual(
            [episode["episode_id"] for episode in serialized],
            [episode.episode_id for episode in expected],
        )

    def test_benchmark_numeric_and_discrete_paths_equal_public_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            events_path, truth_path = write_suite(build_suite("synthetic_holdout"), Path(directory))
            by_scenario, manifest = load_inputs(events_path, truth_path)
        benchmark = run_variant(by_scenario, manifest, VARIANTS["baseline"])
        benchmark_ids = {
            (episode.metadata["scenario_id"], episode.episode_id)
            for episode in benchmark
            if episode.metadata["scenario_id"]
        }
        pipeline_ids = set()
        for scenario in manifest["scenarios"]:
            if scenario["detector_applicability"] not in {"numeric", "discrete"}:
                continue
            for episode in evaluate_channel(by_scenario[scenario["scenario_id"]]).detector_results:
                pipeline_ids.add((scenario["scenario_id"], episode.episode_id))
        self.assertTrue(pipeline_ids.issubset(benchmark_ids))


if __name__ == "__main__":
    unittest.main()
