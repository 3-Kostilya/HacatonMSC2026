"""Past-only R5 anomaly features for the fixed R3 conditional population.

Cluster numbers are diagnostics, not ordinal model features. HDBSCAN has no
out-of-sample prediction API in scikit-learn; its centroid-distance projection
below is explicitly a proxy, fitted on past examples only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


R5_VERSION = "r5-a-anomaly-features-v1"
INPUT_COLUMNS = (
    "last_observation_age_seconds",
    "event_count_1h", "event_count_6h", "event_count_24h", "event_count_168h",
    "alarm_count_1h", "alarm_count_6h", "alarm_count_24h", "alarm_count_168h",
    "state_transitions_1h", "state_transitions_6h", "state_transitions_24h",
    "state_transitions_168h", "technical_message_count_1h",
    "technical_message_count_24h", "registered_fault_text_count_1h",
    "registered_fault_text_count_24h", "normal_message_count_1h",
    "normal_message_count_24h", "unknown_state_count_24h",
    "completed_episode_count_168h",
)
STAT_COLUMNS = (
    "r5_event_burst_z_1h", "r5_alarm_burst_z_1h",
    "r5_state_transition_burst_z_6h", "r5_event_change_z_6h",
)
MODEL_COLUMNS = (
    "r5_if_anomaly_score", "r5_kmeans_distance", "r5_hdbscan_centroid_distance",
)


def _positive(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    return np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def _contrast(recent: np.ndarray, total: np.ndarray, recent_hours: int, total_hours: int) -> np.ndarray:
    """One-sided rate shift against the *older, disjoint* part of a past window."""

    older = np.maximum(total - recent, 0.0)
    expected = older * recent_hours / (total_hours - recent_hours)
    return (recent - expected) / np.sqrt(expected + 1.0)


def statistical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Window contrasts use only events at or before each prediction time."""

    result = pd.DataFrame(index=frame.index)
    for output, base, short, long, short_h, long_h in (
        ("r5_event_burst_z_1h", "event_count", "1h", "24h", 1, 24),
        ("r5_alarm_burst_z_1h", "alarm_count", "1h", "24h", 1, 24),
        ("r5_state_transition_burst_z_6h", "state_transitions", "6h", "168h", 6, 168),
        ("r5_event_change_z_6h", "event_count", "6h", "168h", 6, 168),
    ):
        recent = _positive(frame, f"{base}_{short}")
        total = _positive(frame, f"{base}_{long}")
        result[output] = _contrast(recent, total, short_h, long_h).astype("float32")
    return result


def model_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Fixed, non-fitted transform; missingness is visible to every model."""

    columns = []
    for name in INPUT_COLUMNS:
        raw = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        missing = ~np.isfinite(raw)
        values = np.log1p(np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), 0, 1e9))
        columns.extend((values, missing.astype(np.float64)))
    return np.column_stack(columns).astype(np.float32)


@dataclass
class TypeModels:
    sensor_type: str
    fit_end_at: pd.Timestamp
    fit_rows: int
    fit_channels: int
    scaler: RobustScaler
    forest: IsolationForest
    kmeans: MiniBatchKMeans
    hdbscan: HDBSCAN
    hdbscan_centroids: np.ndarray


def fit_type_models(
    frame: pd.DataFrame, *, fit_end_at: pd.Timestamp, min_rows: int = 80,
    min_channels: int = 3, seed: int = 42,
) -> TypeModels | None:
    """Fit one compatible type, never crossing the requested temporal cutoff."""

    if frame.empty:
        return None
    if frame["sensor_type"].nunique(dropna=False) != 1:
        raise ValueError("R5 models require exactly one sensor type")
    if (frame["prediction_time"] >= fit_end_at).any():
        raise ValueError("R5 fit includes the score period or future")
    channels = frame["channel_id"].nunique()
    if len(frame) < min_rows or channels < min_channels:
        return None
    X = model_matrix(frame)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(X)
    scaled = np.clip(scaler.transform(X), -20, 20).astype(np.float32)
    forest = IsolationForest(n_estimators=64, max_samples=min(256, len(frame)),
                             random_state=seed, n_jobs=1).fit(scaled)
    k = min(6, max(2, int(math.sqrt(len(frame) / 100))))
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=256,
                             n_init=3).fit(scaled)
    hdbscan = HDBSCAN(min_cluster_size=max(20, min(60, len(frame) // 20)),
                      min_samples=10, store_centers="centroid", copy=True).fit(scaled)
    centroids = np.asarray(hdbscan.centroids_, dtype=np.float32)
    return TypeModels(str(frame["sensor_type"].iloc[0]), fit_end_at, len(frame), channels,
                      scaler, forest, kmeans, hdbscan, centroids)


def score_type_models(frame: pd.DataFrame, model: TypeModels | None) -> pd.DataFrame:
    """Score unseen rows; mode IDs are diagnostics and scoped to this fitted model."""

    result = statistical_features(frame)
    for name in MODEL_COLUMNS:
        result[name] = np.full(len(frame), np.nan, dtype=np.float32)
    result["r5_kmeans_mode_id"] = np.full(len(frame), -1, dtype=np.int16)
    result["r5_hdbscan_proxy_mode_id"] = np.full(len(frame), -1, dtype=np.int16)
    if model is None or frame.empty:
        return result
    if frame["sensor_type"].nunique(dropna=False) != 1 or frame["sensor_type"].iloc[0] != model.sensor_type:
        raise ValueError("R5 model sensor type differs from scored rows")
    if (frame["prediction_time"] < model.fit_end_at).any():
        raise ValueError("R5 score precedes fitted-model cutoff")
    scaled = np.clip(model.scaler.transform(model_matrix(frame)), -20, 20).astype(np.float32)
    result["r5_if_anomaly_score"] = -model.forest.score_samples(scaled).astype(np.float32)
    distances = model.kmeans.transform(scaled)
    result["r5_kmeans_distance"] = distances.min(axis=1).astype(np.float32)
    result["r5_kmeans_mode_id"] = distances.argmin(axis=1).astype(np.int16)
    if model.hdbscan_centroids.size:
        # A descriptive projection, not HDBSCAN's exact density membership.
        distances = np.linalg.norm(scaled[:, None, :] - model.hdbscan_centroids[None, :, :], axis=2)
        result["r5_hdbscan_centroid_distance"] = distances.min(axis=1).astype(np.float32)
        result["r5_hdbscan_proxy_mode_id"] = distances.argmin(axis=1).astype(np.int16)
    return result
