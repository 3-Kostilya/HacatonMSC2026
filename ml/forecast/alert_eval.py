"""Evaluate distinct warnings and episode matches on a fixed validation set."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd


SCORE_COLUMNS = ("rule_score", "logistic_score", "catboost_score")
COOLDOWN = timedelta(hours=24)


def evaluate_alerts(frame: pd.DataFrame, score_column: str, threshold: float,
                    *, channel_days: int, cooldown: timedelta = COOLDOWN
                    ) -> tuple[dict, pd.DataFrame]:
    """Apply per-channel cooldown before matching one warning per future episode.

    Every selected row is a possible warning. An earlier false warning can
    suppress a later true one. Repeated warnings for an already matched episode
    count as operationally unmatched, never as extra true positives.
    """
    if score_column not in SCORE_COLUMNS or not 0 < cooldown.total_seconds():
        raise ValueError("invalid score column or cooldown")
    if channel_days <= 0 or pd.isna(threshold):
        raise ValueError("channel-days and threshold must be valid")
    required = {"channel_id", "prediction_time", "sensor_type", "target",
                "target_episode_id", "label_available_at", score_column}
    if not required <= set(frame.columns):
        raise ValueError("validation predictions lack required columns")
    positives = frame.loc[frame["target"] == 1, "target_episode_id"]
    if positives.isna().any():
        raise ValueError("positive rows lack episode IDs")
    truth_count = int(positives.nunique())
    selected = frame.loc[frame[score_column] >= threshold, [
        "channel_id", "prediction_time", "sensor_type", "target",
        "target_episode_id", "label_available_at", score_column,
    ]].sort_values(["channel_id", "prediction_time"], kind="mergesort")
    last_emitted: dict[str, pd.Timestamp] = {}
    matched: set[str] = set()
    emitted: list[dict] = []
    suppressed = 0
    for channel, at, sensor_type, target, episode, onset, score in selected.itertuples(
        index=False, name=None
    ):
        previous = last_emitted.get(channel)
        if previous is not None and at < previous + cooldown:
            suppressed += 1
            continue
        last_emitted[channel] = at
        lead_hours = None
        if target == 1:
            lead_hours = (onset - at).total_seconds() / 3600
            if not 0 < lead_hours <= 24:
                raise ValueError("positive warning has invalid lead time")
            if episode not in matched:
                matched.add(episode)
                outcome = "matched_episode"
            else:
                outcome = "duplicate_episode_warning"
        elif target == 0:
            outcome = "no_target_in_horizon"
        else:
            raise ValueError("non-binary row in accepted validation frame")
        emitted.append({
            "channel_id": channel, "prediction_time": at,
            "sensor_type": sensor_type, "target": int(target),
            "target_episode_id": episode, "label_available_at": onset,
            "score": float(score), "outcome": outcome,
            "lead_hours": lead_hours,
        })
    alerts = pd.DataFrame(emitted, columns=[
        "channel_id", "prediction_time", "sensor_type", "target",
        "target_episode_id", "label_available_at", "score", "outcome",
        "lead_hours",
    ])
    matched_count = len(matched)
    unmatched_count = len(alerts) - matched_count
    lead = alerts.loc[alerts["outcome"] == "matched_episode", "lead_hours"]
    precision = matched_count / len(alerts) if len(alerts) else 0.0
    recall = matched_count / truth_count if truth_count else 0.0
    result = {
        "score_column": score_column,
        "threshold": float(threshold),
        "cooldown_hours": cooldown.total_seconds() / 3600,
        "eligible_positive_episodes": truth_count,
        "emitted_warnings": len(alerts),
        "suppressed_positive_score_rows": suppressed,
        "matched_episodes": matched_count,
        "unmatched_warnings": unmatched_count,
        "duplicate_episode_warnings": int((alerts["outcome"] == "duplicate_episode_warning").sum()),
        "episode_precision": precision,
        "episode_recall": recall,
        "episode_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "unmatched_warnings_per_1000_channel_days": unmatched_count * 1000 / channel_days,
        "median_lead_hours": float(lead.median()) if not lead.empty else None,
    }
    return result, alerts
