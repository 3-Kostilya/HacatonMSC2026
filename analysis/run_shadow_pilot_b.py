"""Record frozen R6 shadow decisions from A's past-only hourly input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.train_r4_discrete_baselines import sha256
from ml.forecast.r6_rule import TERMS
from ml.forecast.shadow_pilot import (
    ShadowPolicy, ShadowState, forbidden_input_fields, run_shadow_batch,
    summarize_shadow_batch,
)


INPUT_FIELDS = frozenset({
    "channel_id", "sensor_type", "prediction_time", "admission_status",
    "admission_reason", "history_through", "admission_through", *TERMS,
})
DECISION_SCHEMA = pa.schema([
    pa.field("policy_version", pa.string()),
    pa.field("freeze_sha256", pa.string()),
    pa.field("channel_id", pa.string()),
    pa.field("prediction_time", pa.string()),
    pa.field("sensor_type", pa.string()),
    pa.field("admission_status", pa.string()),
    pa.field("prediction_status", pa.string()),
    pa.field("unavailable_reason", pa.string()),
    pa.field("rule_score", pa.float64()),
    pa.field("threshold_crossed", pa.bool_()),
    pa.field("shadow_warning", pa.bool_()),
    pa.field("warning_reason", pa.string()),
    pa.field("score_contributions", pa.struct([
        pa.field(name, pa.float64()) for name in TERMS
    ])),
    pa.field("delivery_mode", pa.string()),
    pa.field("automatic_action_taken", pa.bool_()),
])


def run(*, input_path: Path, freeze_path: Path, output_dir: Path,
        checkpoint_in: Path | None = None,
        expected_channel_hours: int | None = None) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source = pq.ParquetFile(input_path)
    absent = INPUT_FIELDS - set(source.schema_arrow.names)
    leaked = forbidden_input_fields(source.schema_arrow.names)
    if absent or leaked:
        raise ValueError(f"shadow input schema differs: absent={sorted(absent)}, "
                         f"future_fields={sorted(leaked)}")
    policy = ShadowPolicy.from_freeze(freeze_path)
    state = (ShadowState.restore(json.loads(checkpoint_in.read_text(encoding="utf-8")),
                                 policy) if checkpoint_in else ShadowState())
    output_dir.mkdir(parents=True)
    decisions_path = output_dir / "shadow_decisions.parquet"
    with pq.ParquetWriter(decisions_path, DECISION_SCHEMA, compression="zstd") as writer:
        for batch in source.iter_batches(batch_size=8192, columns=sorted(INPUT_FIELDS)):
            decisions = run_shadow_batch(batch.to_pylist(), state, policy)
            writer.write_table(pa.Table.from_pylist(decisions, schema=DECISION_SCHEMA))

    def saved_rows():
        for batch in pq.ParquetFile(decisions_path).iter_batches(batch_size=8192):
            yield from batch.to_pylist()

    summary = summarize_shadow_batch(saved_rows(),
                                     expected_channel_hours=expected_channel_hours)
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(state.checkpoint(policy), ensure_ascii=False,
                                          indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": "r6-b-shadow-batch-report-v1",
        "status": "complete_record_only",
        "source_input_sha256": sha256(input_path),
        "source_freeze_sha256": policy.freeze_sha256,
        "source_checkpoint_sha256": sha256(checkpoint_in) if checkpoint_in else None,
        "summary": summary,
        "limitations": [
            "A must independently verify past-only admission and feature evidence.",
            "Historical replay is not a new independent test of model quality.",
            "No customer notification or automatic action occurred.",
            "False-warning rate requires later target labels and archive review.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({
        "schema_version": report["schema_version"],
        "status": report["status"],
        "source_freeze_sha256": policy.freeze_sha256,
        "decisions_sha256": sha256(decisions_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "report_sha256": sha256(report_path),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--freeze", type=Path,
                        default=Path("ml/r6_frozen_rule_v1.json"))
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--expected-channel-hours", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(input_path=args.input, freeze_path=args.freeze,
                 output_dir=args.output_dir, checkpoint_in=args.checkpoint_in,
                 expected_channel_hours=args.expected_channel_hours)
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
