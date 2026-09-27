"""Fast research evaluation with the canonical 24h warning and episode semantics.

All recall/F1 numbers use the caller's complete assigned episode denominator,
including unavailable episodes. Per-type thresholds and score combinations are
research options; choosing them on an open year is not an independent result.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


DAY_NS = 24 * 3600 * 10**9
EVALUATION_VERSION = "full-b3-canonical-native-score-precision-v2"
REQUIRED = {"channel_id", "prediction_time", "sensor_type", "target",
            "target_episode_id", "label_available_at"}


class PreparedEvaluation:
    """Presort and encode a frame once for many threshold/model evaluations."""

    def __init__(self, frame: pd.DataFrame, full_episode_count: int,
                 channel_days: int | None = None):
        if not REQUIRED <= set(frame):
            raise ValueError(f"evaluation lacks columns: {sorted(REQUIRED - set(frame))}")
        if full_episode_count < 0:
            raise ValueError("full episode count cannot be negative")
        if frame.channel_id.isna().any() or frame.prediction_time.isna().any():
            raise ValueError("channel and prediction time must be present")
        if not frame.target.isin([0, 1]).all():
            raise ValueError("evaluation accepts binary assigned rows only")
        positive = frame.target.eq(1)
        if frame.loc[positive, "target_episode_id"].isna().any():
            raise ValueError("positive rows lack episode IDs")
        self.available_episodes = int(frame.loc[positive, "target_episode_id"].nunique())
        if full_episode_count < self.available_episodes:
            raise ValueError("full denominator is smaller than available episodes")
        self.full_episode_count = int(full_episode_count)
        # Stable sorting matches the production evaluator even for equal timestamps.
        self.frame = frame.sort_values(["channel_id", "prediction_time"], kind="mergesort")
        self.channels = pd.factorize(self.frame.channel_id, sort=False)[0]
        self.times = pd.to_datetime(self.frame.prediction_time).to_numpy(
            dtype="datetime64[ns]").astype(np.int64)
        self.targets = self.frame.target.to_numpy(dtype=np.int8)
        self.episodes, self.episode_ids = pd.factorize(
            self.frame.target_episode_id, sort=False)
        self.onsets = pd.to_datetime(self.frame.label_available_at).to_numpy(
            dtype="datetime64[ns]").astype(np.int64)
        self.types = self.frame.sensor_type.fillna("<unknown>").astype(str).to_numpy()
        if channel_days is None:
            dates = self.times // DAY_NS
            channel_days = (int(1 + np.count_nonzero(
                (self.channels[1:] != self.channels[:-1]) | (dates[1:] != dates[:-1])))
                if len(frame) else 0)
        if channel_days < 0 or (len(frame) and channel_days == 0):
            raise ValueError("channel-days must be positive for a nonempty frame")
        self.channel_days = int(channel_days)

    def evaluate(self, score_column: str, threshold: float | Mapping[str, float]) -> dict:
        scores = self.frame[score_column].to_numpy()
        if isinstance(threshold, Mapping):
            limits = {str(key): float(value) for key, value in threshold.items()}
            absent = set(self.types) - set(limits)
            if absent:
                raise ValueError(f"per-type thresholds lack types: {sorted(absent)}")
            if any(np.isnan(value) for value in limits.values()):
                raise ValueError("threshold cannot be NaN")
            limit_dtype = scores.dtype if scores.dtype.kind == "f" else np.float64
            selected = np.flatnonzero(scores >= np.array(
                [limits[x] for x in self.types], dtype=limit_dtype))
            threshold_record = limits
        else:
            if np.isnan(threshold):
                raise ValueError("threshold cannot be NaN")
            threshold_record = float(threshold)
            # pandas intentionally compares float32 scores in their native
            # precision. Promoting to float64 changes exact threshold ties.
            selected = np.flatnonzero(self.frame[score_column].ge(threshold).to_numpy())
        last_channel = -1
        last_time = 0
        matched: set[int] = set()
        leads = []
        warnings = suppressed = duplicate = 0
        by_type: dict[str, dict] = {}
        for i in selected:
            channel, at = self.channels[i], self.times[i]
            if channel == last_channel and at - last_time < DAY_NS:
                suppressed += 1
                continue
            last_channel, last_time = channel, at
            warnings += 1
            kind = self.types[i]
            type_stats = by_type.setdefault(kind, {"emitted_warnings": 0,
                                                    "matched_episodes": 0})
            type_stats["emitted_warnings"] += 1
            if self.targets[i]:
                lead = (self.onsets[i] - at) / (3600 * 10**9)
                if not 0 < lead <= 24:
                    raise ValueError("positive warning has invalid lead time")
                episode = int(self.episodes[i])
                if episode not in matched:
                    matched.add(episode)
                    leads.append(lead)
                    type_stats["matched_episodes"] += 1
                else:
                    duplicate += 1
        tp = len(matched)
        precision = tp / warnings if warnings else 0.0
        recall = tp / self.full_episode_count if self.full_episode_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        for stats in by_type.values():
            stats["episode_precision"] = stats["matched_episodes"] / stats["emitted_warnings"]
        return {
            "evaluation_version": EVALUATION_VERSION,
            "score_column": score_column, "threshold": threshold_record,
            "cooldown_hours": 24.0,
            "eligible_positive_episodes": self.available_episodes,
            "full_episode_count": self.full_episode_count,
            "emitted_warnings": warnings,
            "suppressed_positive_score_rows": suppressed,
            "matched_episodes": tp, "unmatched_warnings": warnings - tp,
            "duplicate_episode_warnings": duplicate,
            "episode_precision": precision, "episode_recall": recall,
            "full_episode_recall": recall, "episode_f1": f1, "full_episode_f1": f1,
            "available_episode_recall": tp / self.available_episodes
                if self.available_episodes else 0.0,
            "unmatched_warnings_per_1000_channel_days": (warnings - tp) * 1000
                / self.channel_days if self.channel_days else 0.0,
            "median_lead_hours": float(np.median(leads)) if leads else None,
            "channel_days": self.channel_days, "by_type": by_type,
            "matched_episode_ids": self.episode_ids.take(sorted(matched)).tolist(),
        }


def evaluate(frame: pd.DataFrame, score_column: str,
             threshold: float | Mapping[str, float], full_episode_count: int,
             channel_days: int | None = None) -> dict:
    return PreparedEvaluation(frame, full_episode_count, channel_days).evaluate(
        score_column, threshold)


def threshold_grid(scores: np.ndarray, points: int = 61) -> list[float]:
    """A score-only grid concentrating effort on sparse warning candidates."""
    if not 40 <= points <= 80:
        raise ValueError("threshold search requires 40 to 80 points")
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [float("inf")]
    tail = 1 - np.geomspace(0.20, 0.00001, points - 1)
    values = np.quantile(finite, tail).tolist()
    # A float64 nextafter(max_float32) rounds back to max under pandas' native
    # float32 comparison. A margin above float32 precision gives a true silent
    # policy in both evaluators and remains portable finite JSON.
    values.append(float(finite.max() + max(1.0, abs(finite.max())) * 1e-6))
    return sorted(set(float(value) for value in values))


def search_thresholds(frame: pd.DataFrame | PreparedEvaluation, score_column: str,
                      full_episode_count: int | None = None,
                      channel_days: int | None = None, points: int = 61) -> list[dict]:
    if isinstance(frame, PreparedEvaluation):
        prepared = frame
    else:
        if full_episode_count is None:
            raise ValueError("full episode denominator is required")
        prepared = PreparedEvaluation(frame, full_episode_count, channel_days)
    return [prepared.evaluate(score_column, threshold) for threshold in
            threshold_grid(prepared.frame[score_column].to_numpy(), points)]


def combine_scores(frame: pd.DataFrame, columns: list[str], *, method: str = "mean",
                   weights: list[float] | None = None) -> np.ndarray:
    """Exploratory row-wise fusion. Freeze method/weights before held-out use."""
    matrix = frame[columns].to_numpy(dtype=float)
    if not columns:
        raise ValueError("score columns are required")
    if method == "mean":
        return np.average(matrix, axis=1, weights=weights)
    if weights is not None:
        raise ValueError("weights only apply to mean fusion")
    if method == "max":
        return matrix.max(axis=1)
    if method == "min":
        return matrix.min(axis=1)
    if method == "rank_mean":
        return frame[columns].rank(pct=True).mean(axis=1).to_numpy()
    raise ValueError(f"unknown combination method: {method}")
