"""Run the trained research CatBoost on an already-precomputed causal feature Parquet.

This is the integration bridge for the current repository state:

prepared Q2/Q3 feature rows -> research CatBoost + R6 score preview -> FastAPI

It intentionally does NOT pretend to build the 121 Q2/Q3 features from raw events.
The current trained service bundle requires those features to be precomputed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from ml.forecast.r6_rule import TERMS, predict_rule
from ml.forecast.shadow_pilot import ShadowPolicy
from ml.service_candidate.loader import ResearchRiskModel
from ml_runtime.backend_client import check_backend, post_records


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "service-model-journal-failure-20260928-v1"
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
PREVIEW_POLICY_VERSION = "r6-integration-preview-v1"
RESEARCH_SCORE_KIND = "uncalibrated_research"


def _resolve_bundle(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Model bundle directory not found: {candidate}")

    env = os.getenv("MODEL_BUNDLE_DIR")
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            ROOT / "artifacts" / BUNDLE_NAME,
            ROOT / "ML" / "artifacts" / BUNDLE_NAME,
            ROOT / BUNDLE_NAME,
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "model.cbm").is_file() and (candidate / "model_metadata.json").is_file():
            return candidate

    rendered = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find the trained model bundle. Pass --model or set MODEL_BUNDLE_DIR.\n"
        f"Checked:\n{rendered}"
    )


def _unique(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


def _read_sample(
    parquet_path: Path,
    *,
    base_features: list[str],
    limit: int,
) -> pd.DataFrame:
    """Read at most one prepared row per channel without loading a huge file."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    parquet = pq.ParquetFile(parquet_path)
    available = set(parquet.schema_arrow.names)
    required = _unique(["channel_id", "prediction_time", *base_features])
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(
            "The selected Parquet is not the expected prepared Q2/Q3 feature table. "
            f"Missing columns: {missing}"
        )

    # Keep only one row per channel in memory. This works even when the Parquet
    # contains millions of rows.
    selected: dict[str, dict[str, Any]] = {}
    columns = required
    for batch in parquet.iter_batches(batch_size=4096, columns=columns):
        part = batch.to_pandas()
        if part.empty:
            continue
        part["channel_id"] = part["channel_id"].astype("string")
        part["prediction_time"] = pd.to_datetime(part["prediction_time"], errors="coerce")
        part = part[part["channel_id"].notna() & part["prediction_time"].notna()]
        if part.empty:
            continue

        for _, row in part.drop_duplicates("channel_id", keep="last").iterrows():
            channel = str(row["channel_id"])
            if channel in selected or len(selected) < limit:
                selected[channel] = row.to_dict()
        if len(selected) >= limit:
            break

    if not selected:
        raise ValueError("No usable rows were found in the prepared feature Parquet")

    frame = pd.DataFrame(selected.values()).reset_index(drop=True)
    frame["prediction_time"] = pd.to_datetime(frame["prediction_time"], errors="raise")
    frame = frame.sort_values(["channel_id", "prediction_time"]).reset_index(drop=True)

    # The training/service packaging code scores prepared fold rows as explicitly
    # eligible. For live data this must later come from the real admission pipeline.
    frame["admission_status"] = "eligible"
    return frame


def _optional_number(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    number = _optional_number(value)
    if number is None:
        return None
    if number < 0 or not number.is_integer():
        return None
    return int(number)


def _build_forecasts(
    frame: pd.DataFrame,
    *,
    model: ResearchRiskModel,
    policy: ShadowPolicy,
) -> list[dict[str, Any]]:
    base_features = model.metadata["base_feature_names"]
    service_input = frame[["channel_id", "prediction_time", "admission_status", *base_features]].copy()
    research = model.score(service_input).reset_index(drop=True)

    rule = predict_rule(
        frame,
        eligibility_status=frame["admission_status"],
        threshold=policy.threshold,
    ).reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for index, row in frame.reset_index(drop=True).iterrows():
        prediction_time = pd.Timestamp(row["prediction_time"]).to_pydatetime().replace(tzinfo=None)
        research_row = research.iloc[index]
        rule_row = rule.iloc[index]

        counts = {name: _optional_int(row.get(name)) for name in TERMS}
        contributions = None
        if all(value is not None for value in counts.values()):
            contributions = {
                name: float(TERMS[name] * counts[name])
                for name in TERMS
            }

        score = _optional_number(rule_row.get("rule_score"))
        crossed = None if score is None else bool(score >= policy.threshold)
        if score is None:
            prediction_status = "not_available"
            unavailable_reason = "missing_or_invalid_rule_input"
            warning_reason = "no_prediction"
        else:
            prediction_status = "scored"
            unavailable_reason = None
            warning_reason = (
                "threshold_crossed_preview_only"
                if crossed
                else "below_frozen_threshold"
            )

        records.append(
            {
                "policy_version": PREVIEW_POLICY_VERSION,
                "freeze_sha256": policy.freeze_sha256,
                "channel_id": str(row["channel_id"]),
                "prediction_time": prediction_time.isoformat(),
                "sensor_type": str(row["sensor_type"]),
                "admission_status": "eligible",
                "admission_reason": None,
                "prediction_status": prediction_status,
                "unavailable_reason": unavailable_reason,
                "rule_score": score,
                "threshold": policy.threshold,
                "threshold_crossed": crossed,
                # Important: the prepared training fold does not contain the
                # persisted shadow state/evidence-cutoff contract required to
                # issue a real shadow warning. Never manufacture one here.
                "shadow_warning": False,
                "warning_reason": warning_reason,
                "score_contributions": contributions,
                "delivery_mode": "record_only",
                "automatic_action_taken": False,
                "history_through": None,
                "admission_through": None,
                "registered_fault_text_count_24h": counts.get(
                    "registered_fault_text_count_24h"
                ),
                "registered_fault_text_count_168h": counts.get(
                    "registered_fault_text_count_168h"
                ),
                "completed_episode_count_168h": counts.get(
                    "completed_episode_count_168h"
                ),
                "technical_message_count_24h": counts.get(
                    "technical_message_count_24h"
                ),
                "research_score": _optional_number(research_row.get("risk_score")),
                "research_prediction_status": str(research_row["prediction_status"]),
                "research_model_version": str(model.metadata["schema_version"]),
                "research_score_kind": RESEARCH_SCORE_KIND,
            }
        )
    return records


def run(
    *,
    model_dir: Path,
    features_path: Path,
    backend_url: str,
    limit: int,
    batch_size: int,
) -> None:
    print(f"Model bundle: {model_dir}")
    print(f"Prepared features: {features_path}")
    print(f"Backend: {backend_url}")

    check_backend(backend_url)
    print("Backend health: OK")

    model = ResearchRiskModel(model_dir)
    print(f"Research model: {model.metadata['schema_version']}")

    policy = ShadowPolicy.from_freeze(ROOT / "ml" / "r6_frozen_rule_v1.json")
    frame = _read_sample(
        features_path,
        base_features=model.metadata["base_feature_names"],
        limit=limit,
    )
    print(f"Prepared rows selected: {len(frame)}")

    records = _build_forecasts(frame, model=model, policy=policy)
    stored = post_records(
        backend_url,
        "/api/ml/forecasts",
        records,
        batch_size=batch_size,
    )
    research_scored = sum(row["research_score"] is not None for row in records)
    rule_scored = sum(row["rule_score"] is not None for row in records)
    threshold_crossed = sum(row["threshold_crossed"] is True for row in records)

    print(f"Stored forecasts: {stored}")
    print(f"Research CatBoost scores: {research_scored}")
    print(f"R6 scores: {rule_scored}")
    print(f"R6 threshold crossings (preview only): {threshold_crossed}")
    print("Done. Refresh the frontend.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to service-model-journal-failure-20260928-v1 directory",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Prepared Q2/Q3 feature Parquet. Defaults to <model>/data/validation.parquet",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND_URL,
        help="FastAPI base URL",
    )
    parser.add_argument("--limit", type=int, default=50, help="Maximum channels to send")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    model_dir = _resolve_bundle(args.model)
    features_path = (
        args.features.expanduser().resolve()
        if args.features is not None
        else model_dir / "data" / "validation.parquet"
    )
    run(
        model_dir=model_dir,
        features_path=features_path,
        backend_url=args.backend,
        limit=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
