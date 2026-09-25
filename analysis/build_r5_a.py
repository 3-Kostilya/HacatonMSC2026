"""Build causal R5 A anomaly feature blocks from the fixed A3/B R3 packages.

The B candidate file supplies keys and split *only*. Targets are never read.
Run ``python -m analysis.build_r5_a --help`` for source/output arguments.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stage1.features.r5 import (
    INPUT_COLUMNS, MODEL_COLUMNS, R5_VERSION, STAT_COLUMNS,
    fit_type_models, score_type_models,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A3 = ROOT / "output" / "r3-a-full-months-20260924"
DEFAULT_CANDIDATES = ROOT / "output" / "r3-b-conditional-discrete-20260925"
DEFAULT_OUTPUT = ROOT / "output" / "r5-a-anomaly-features-20260925"
KEYS = ["channel_id", "prediction_time"]
SOURCE_COLUMNS = [*KEYS, "sensor_type", *INPUT_COLUMNS]


def _digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hash_.update(block)
    return hash_.hexdigest()


def _month_path(root: Path, month: str, filename: str) -> Path:
    year, number = month.split("-")
    return root / f"year={year}" / f"month={number}" / filename


def _source_manifests(a3: Path, candidates: Path) -> tuple[dict, dict, str, str]:
    a3_file = a3 / "manifest.json"
    candidate_file = candidates / "manifest.json"
    am, cm = json.loads(a3_file.read_text(encoding="utf-8")), json.loads(candidate_file.read_text(encoding="utf-8"))
    a_hash, c_hash = _digest(a3_file), _digest(candidate_file)
    if am["status"] != "complete" or cm["status"] != "complete_conditional_candidates":
        raise ValueError("R5 requires complete A3 and conditional R3 packages")
    if cm["source_a3_manifest_sha256"] != a_hash:
        raise ValueError("B candidate keys were built from another A3 package")
    a_months = {chunk["month"] for chunk in am["chunks"]}
    c_months = {chunk["month"] for chunk in cm["chunks"]}
    if a_months != c_months or len(a_months) != 78:
        raise ValueError("R5 requires the same 78 months in A3 and B candidate packages")
    return am, cm, a_hash, c_hash


def _load_keys(root: Path, month: str) -> pd.DataFrame:
    path = _month_path(root, month, "conditional_discrete_keys.parquet")
    keys = pq.read_table(path, columns=[*KEYS, "sensor_type", "split"]).to_pandas()
    if keys.duplicated(KEYS).any():
        raise ValueError(f"duplicate B candidate key in {month}")
    year = int(month[:4])
    required = "validation" if year == 2025 else "test" if year == 2026 else "train"
    if not keys["split"].eq(required).all():
        raise ValueError(f"unexpected temporal split in {month}")
    return keys


def _load_joined(a3: Path, month: str, keys: pd.DataFrame) -> pd.DataFrame:
    # Read only admitted keys. A full A3 month can be much larger than its
    # R3 candidate subset and needlessly exhaust memory during conversion.
    projection = ", ".join(f'f."{name}"' for name in INPUT_COLUMNS)
    with duckdb.connect(":memory:") as database:
        database.execute("SET threads=2")
        database.register("candidate_keys", keys)
        joined = database.execute(
            f"""SELECT k.channel_id, k.prediction_time,
                       k.sensor_type AS sensor_type_b, k.split,
                       f.sensor_type, {projection}
                FROM candidate_keys AS k
                LEFT JOIN read_parquet(?) AS f
                  USING (channel_id, prediction_time)""",
            [str(_month_path(a3, month, "features.parquet"))],
        ).fetch_df()
    if len(joined) != len(keys) or joined["sensor_type"].isna().any():
        raise ValueError(f"A3/B candidate key mismatch in {month}")
    if not joined["sensor_type_b"].fillna("<null>").eq(joined["sensor_type"].fillna("<null>")).all():
        raise ValueError(f"A3/B sensor type mismatch in {month}")
    return joined.drop(columns=["sensor_type_b"])


def _sample_train(a3: Path, candidates: Path, months: list[str], *,
                  per_channel_month: int = 2, max_per_type_month: int = 24) -> pd.DataFrame:
    """Deterministic channel-balanced bounded sample of the candidate population."""

    chunks = []
    for month in months:
        if int(month[:4]) > 2024:
            continue
        keys = _load_keys(candidates, month)
        if keys.empty:
            continue
        priority = pd.util.hash_pandas_object(keys[KEYS], index=False)
        sampled = (keys.assign(_priority=priority)
                   .sort_values(["channel_id", "_priority"])
                   .groupby("channel_id", sort=False).head(per_channel_month)
                   .drop(columns="_priority"))
        sampled["_priority"] = pd.util.hash_pandas_object(sampled[KEYS], index=False)
        sampled = (sampled.sort_values(["sensor_type", "_priority"])
                   .groupby("sensor_type", sort=False).head(max_per_type_month)
                   .drop(columns="_priority"))
        chunks.append(_load_joined(a3, month, sampled))
    if not chunks:
        raise ValueError("no past training candidates")
    sample = pd.concat(chunks, ignore_index=True)
    if sample.duplicated(KEYS).any():
        raise ValueError("sample has duplicate candidate keys")
    # Both caps are per month, so choosing a later month's sample cannot alter
    # any earlier fold's fit population.
    return sample.reset_index(drop=True)


def _fold(month: str) -> tuple[str, pd.Timestamp, pd.Timestamp] | None:
    if month < "2019-07":
        return None  # 2019 H1 supplies only statistical features and warm-up fit.
    if month[:4] == "2019":
        return "2019-h2", pd.Timestamp("2019-07-01"), pd.Timestamp("2019-07-01")
    year = int(month[:4])
    fit_end = pd.Timestamp(f"{min(year, 2025)}-01-01")
    return (f"before-{min(year, 2025)}", fit_end, fit_end)


def _fit_fold(sample: pd.DataFrame, fold: tuple[str, pd.Timestamp, pd.Timestamp],
              output: Path) -> tuple[dict[str, object], dict]:
    name, fit_end, _ = fold
    subset = sample.loc[sample["prediction_time"] < fit_end]
    models = {}
    summary = {"fit_end_exclusive": fit_end.isoformat(), "sample_rows": len(subset),
               "types": {}}
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for sensor_type, group in subset.groupby("sensor_type", dropna=True, sort=True):
        model = fit_type_models(group, fit_end_at=fit_end)
        if model is None:
            summary["types"][str(sensor_type)] = {"status": "insufficient_history",
                                                   "rows": len(group),
                                                   "channels": group["channel_id"].nunique()}
            continue
        type_id = hashlib.sha256(str(sensor_type).encode("utf-8")).hexdigest()[:16]
        model_path = model_dir / f"{name}-{type_id}.joblib"
        joblib.dump(model, model_path)
        models[str(sensor_type)] = model
        summary["types"][str(sensor_type)] = {
            "status": "fitted", "rows": model.fit_rows, "channels": model.fit_channels,
            "hdbscan_clusters": len(model.hdbscan_centroids),
            "model_file": str(model_path.relative_to(output)).replace("\\", "/"),
            "model_sha256": _digest(model_path),
        }
    return models, summary


def _score_month(joined: pd.DataFrame, models: dict, fold_name: str,
                 prior: dict, month: str) -> pd.DataFrame:
    joined = joined.sort_values(KEYS).reset_index(drop=True)
    for sensor_type, group in joined.groupby("sensor_type", dropna=False, sort=False):
        block = score_type_models(group, models.get(sensor_type))
        for name in block.columns:
            joined.loc[group.index, name] = block[name].to_numpy()
    for name in ("r5_kmeans_mode_changed", "r5_hdbscan_proxy_mode_changed"):
        joined[name] = np.nan
    # A transition is defined only between adjacent *candidate* hours, at most
    # 24 hours apart, under the same frozen model. It does not claim continuity.
    for idx, row in joined.iterrows():
        key = row["channel_id"]
        previous = prior.get(key)
        if previous is not None:
            old_time, old_fold, old_k, old_h = previous
            gap = (row["prediction_time"] - old_time).total_seconds() / 3600
            if old_fold == fold_name and 0 < gap <= 24:
                if row["r5_kmeans_mode_id"] >= 0 and old_k >= 0:
                    joined.at[idx, "r5_kmeans_mode_changed"] = int(row["r5_kmeans_mode_id"] != old_k)
                if row["r5_hdbscan_proxy_mode_id"] >= 0 and old_h >= 0:
                    joined.at[idx, "r5_hdbscan_proxy_mode_changed"] = int(row["r5_hdbscan_proxy_mode_id"] != old_h)
        prior[key] = (row["prediction_time"], fold_name,
                      row["r5_kmeans_mode_id"], row["r5_hdbscan_proxy_mode_id"])
    joined["r5_model_fold"] = fold_name
    columns = [*KEYS, "sensor_type", "r5_model_fold", *STAT_COLUMNS, *MODEL_COLUMNS,
               "r5_kmeans_mode_id", "r5_hdbscan_proxy_mode_id",
               "r5_kmeans_mode_changed", "r5_hdbscan_proxy_mode_changed"]
    return joined[columns]


def build(a3: Path, candidates: Path, output: Path, *,
          score_start: str | None = None, score_end: str | None = None) -> dict:
    am, cm, a_hash, c_hash = _source_manifests(a3, candidates)
    months = sorted(chunk["month"] for chunk in am["chunks"])
    selected = [m for m in months if (score_start is None or m >= score_start)
                and (score_end is None or m <= score_end)]
    if not selected:
        raise ValueError("no score months selected")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"R5 output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sample = _sample_train(a3, candidates, months)
    pq.write_table(pa.Table.from_pandas(sample, preserve_index=False), output / "fit_sample.parquet",
                   compression="zstd")
    fold_cache: dict[str, tuple[dict, dict]] = {}
    prior: dict = {}
    chunks = []
    totals = Counter()
    for month in selected:
        fold = _fold(month)
        fold_name = fold[0] if fold else "warmup-stat-only"
        if fold and fold_name not in fold_cache:
            fold_cache[fold_name] = _fit_fold(sample, fold, output)
        models = fold_cache[fold_name][0] if fold else {}
        keys = _load_keys(candidates, month)
        joined = _load_joined(a3, month, keys)
        scored = _score_month(joined, models, fold_name, prior, month)
        if len(scored) != len(keys) or scored.duplicated(KEYS).any():
            raise ValueError(f"R5 key preservation failed in {month}")
        path = _month_path(output, month, "features.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(scored, preserve_index=False), path, compression="zstd")
        available = int(scored["r5_if_anomaly_score"].notna().sum())
        chunks.append({"month": month, "rows": len(scored), "models_available_rows": available,
                       "features_file": str(path.relative_to(output)).replace("\\", "/"),
                       "features_sha256": _digest(path)})
        totals["rows"] += len(scored)
        totals["models_available_rows"] += available
        print(f"{month}: {len(scored):,} keys, {available:,} model scores", flush=True)
    report = {
        "schema_version": R5_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_a3_manifest_sha256": a_hash,
        "source_candidate_manifest_sha256": c_hash,
        "target_columns_read": False,
        "fit_sample_rows": len(sample),
        "fit_sample_channels": sample["channel_id"].nunique(),
        "fit_sample_sha256": _digest(output / "fit_sample.parquet"),
        "folds": {name: details for name, (_, details) in fold_cache.items()},
        "rows": totals["rows"],
        "model_score_rows": totals["models_available_rows"],
        "chunks": chunks,
        "limitations": [
            "2019 H1 has no fitted model; statistical contrasts only",
            "HDBSCAN scores and modes are nearest-centroid proxies, not exact density predictions",
            "transition features compare adjacent admitted candidate hours only",
            "all outputs remain conditional journal-record features, not physical-failure labels",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a3", type=Path, default=DEFAULT_A3)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--score-start", help="inclusive YYYY-MM pilot range")
    parser.add_argument("--score-end", help="inclusive YYYY-MM pilot range")
    args = parser.parse_args()
    report = build(args.a3, args.candidates, args.output,
                   score_start=args.score_start, score_end=args.score_end)
    print(json.dumps({"rows": report["rows"], "model_score_rows": report["model_score_rows"],
                      "months": len(report["chunks"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
