"""Causal hourly features and conservatively observed 24-hour fault labels."""

import numpy as np
import pandas as pd

FAULT = "Неисправен"
SMOKE = "Обнаружен дым"
NONFAULT = frozenset(["Норма", "Дыма нет", SMOKE])
AMBIGUOUS = "AMBIGUOUS"
TARGET = "sensor_fault_onset_24h"
HOUR_NS = 3_600_000_000_000
HORIZON_HOURS = 24
MAX_GAP_HOURS = 24
CAT_FEATURES = ["current_state"]
META_COLUMNS = ["channel_id", "timestamp", TARGET, "next_onset", "label_reason", "split"]


def canonicalize(events):
    """Do not invent an order for conflicting states at the same timestamp."""
    required = {"channel_id", "timestamp", "state", "alarm"}
    if not required.issubset(events.columns):
        raise ValueError(f"Required columns: {sorted(required)}")
    frame = events[list(required)].copy()
    if frame.isna().any().any():
        raise ValueError("Missing channel, timestamp, state or alarm")
    frame["channel_id"] = frame.channel_id.astype(str)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, errors="raise")
    if not pd.api.types.is_bool_dtype(frame.alarm):
        alarm = frame.alarm.astype(str).str.lower()
        if not alarm.isin(["true", "false", "t", "f", "1", "0"]).all():
            raise ValueError("Unrecognized alarm value")
        frame["alarm"] = alarm.isin(["true", "t", "1"])
    frame = frame.drop_duplicates()
    grouped = frame.groupby(["channel_id", "timestamp"], sort=True)
    canonical = grouped.agg(
        state=("state", "first"), alarm=("alarm", "max"), state_count=("state", "nunique")
    ).reset_index()
    canonical.loc[canonical.state_count.gt(1), "state"] = AMBIGUOUS
    return canonical.sort_values(["channel_id", "timestamp"]).reset_index(drop=True)


def timeline(group, max_gap_hours=MAX_GAP_HOURS):
    times = group.timestamp.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    states = group.state.to_numpy(dtype=str)
    known = np.isin(states, list(NONFAULT) + [FAULT])
    nonfault = np.isin(states, list(NONFAULT))
    fault = states == FAULT
    gaps = np.r_[np.inf, np.diff(times) / HOUR_NS]
    previous_known = np.r_[False, known[:-1]]
    breaks = (gaps > max_gap_hours) | ~known | ~previous_known
    segments = np.cumsum(breaks)
    previous_nonfault = np.r_[False, nonfault[:-1]]
    onset = fault & previous_nonfault & ~breaks
    changed = np.r_[True, states[1:] != states[:-1]]
    return {
        "times": times,
        "states": states,
        "known": known,
        "nonfault": nonfault,
        "fault": fault,
        "gaps": gaps,
        "segments": segments,
        "onset": onset,
        "changed": changed,
    }


def window_sum(values, left, right):
    prefix = np.r_[0.0, np.cumsum(np.asarray(values, dtype=float))]
    return prefix[right] - prefix[left]


def features_at(group, prediction_times, max_gap_hours=MAX_GAP_HOURS):
    """Only events with timestamp <= prediction time contribute to any feature."""
    line = timeline(group, max_gap_hours)
    times, states = line["times"], line["states"]
    queries = pd.DatetimeIndex(prediction_times).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    indices = np.searchsorted(times, queries, side="right") - 1
    valid = indices >= 0
    queries, indices = queries[valid], indices[valid]
    if not len(queries):
        return pd.DataFrame()
    ages = (queries - times[indices]) / HOUR_NS
    eligible = line["nonfault"][indices] & (ages <= max_gap_hours)
    eligible &= queries - times[0] >= 24 * HOUR_NS
    queries, indices, ages = queries[eligible], indices[eligible], ages[eligible]
    if not len(queries):
        return pd.DataFrame()
    dates = pd.DatetimeIndex(queries.astype("datetime64[ns]"))
    previous_change = np.maximum.accumulate(np.where(line["changed"], np.arange(len(times)), 0))
    features = {
        "channel_id": np.repeat(str(group.channel_id.iloc[0]), len(queries)),
        "timestamp": dates,
        "current_state": states[indices],
        "current_alarm": group.alarm.to_numpy(dtype=np.int8)[indices],
        "hours_since_event": ages,
        "hours_since_state_change": (queries - times[previous_change[indices]]) / HOUR_NS,
        "history_hours": (queries - times[0]) / HOUR_NS,
        "last_gap_hours": np.minimum(line["gaps"][indices], 24 * 365),
        "hour_sin": np.sin(2 * np.pi * dates.hour / 24),
        "hour_cos": np.cos(2 * np.pi * dates.hour / 24),
        "weekday": dates.dayofweek,
        "month": dates.month,
    }
    fault_indices = np.flatnonzero(line["fault"])
    fault_positions = np.searchsorted(times[fault_indices], queries, side="right") - 1
    since_fault = np.full(len(queries), -1.0)
    has_fault = fault_positions >= 0
    since_fault[has_fault] = (
        queries[has_fault] - times[fault_indices[fault_positions[has_fault]]]
    ) / HOUR_NS
    features["hours_since_fault"] = since_fault
    signals = {
        "events": np.ones(len(times)),
        "alarms": group.alarm.to_numpy(dtype=int),
        "faults": line["fault"],
        "onsets": line["onset"],
        "smoke": states == SMOKE,
        "unknown": ~line["known"],
        "conflicts": states == AMBIGUOUS,
        "transitions": line["changed"],
    }
    finite_gaps = np.where(np.isfinite(line["gaps"]), line["gaps"], 0)
    right = indices + 1
    for hours in (1, 24, 168):
        left = np.searchsorted(times, queries - hours * HOUR_NS, side="right")
        for name, values in signals.items():
            features[f"{name}_{hours}h"] = window_sum(values, left, right)
        count = features[f"events_{hours}h"]
        features[f"alarm_share_{hours}h"] = features[f"alarms_{hours}h"] / np.maximum(count, 1)
        features[f"unknown_share_{hours}h"] = features[f"unknown_{hours}h"] / np.maximum(count, 1)
        features[f"gap_mean_{hours}h"] = window_sum(finite_gaps, left, right) / np.maximum(count, 1)
    result = pd.DataFrame(features)
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].astype("float32")
    return result


def label_at(group, prediction_times, max_gap_hours=MAX_GAP_HOURS):
    """Future information is confined to labels and audit metadata."""
    line = timeline(group, max_gap_hours)
    times, segments = line["times"], line["segments"]
    queries = pd.DatetimeIndex(prediction_times).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    indices = np.searchsorted(times, queries, side="right") - 1
    if (indices < 0).any():
        raise ValueError("Prediction precedes first observation")
    end_indices = np.r_[np.flatnonzero(np.diff(segments)), len(segments) - 1]
    segment_ends = dict(zip(segments[end_indices], times[end_indices]))
    observed_until = np.array([segment_ends[s] for s in segments[indices]], dtype=np.int64)
    onsets = np.flatnonzero(line["onset"])
    next_positions = np.searchsorted(times[onsets], queries, side="right")
    has_next = next_positions < len(onsets)
    next_time = np.full(len(queries), np.iinfo(np.int64).max, dtype=np.int64)
    same_segment = np.zeros(len(queries), dtype=bool)
    if len(onsets):
        next_indices = onsets[next_positions[has_next]]
        next_time[has_next] = times[next_indices]
        same_segment[has_next] = segments[next_indices] == segments[indices[has_next]]
    horizon_end = queries + HORIZON_HOURS * HOUR_NS
    eligible = line["nonfault"][indices] & ((queries - times[indices]) <= max_gap_hours * HOUR_NS)
    # Apply the same observation requirement to both classes to avoid selecting
    # short positive windows while demanding complete follow-up for negatives.
    complete_window = horizon_end <= observed_until
    positive = eligible & complete_window & same_segment & (next_time <= horizon_end)
    negative = eligible & complete_window & ~positive
    labels = np.full(len(queries), -1, dtype=np.int8)
    labels[negative], labels[positive] = 0, 1
    next_dates = np.full(len(queries), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    next_dates[positive] = next_time[positive].astype("datetime64[ns]")
    return pd.DataFrame(
        {
            TARGET: labels,
            "next_onset": next_dates,
            "label_reason": np.where(
                positive, "observed_onset", np.where(negative, "observed_24h_no_onset", "censored")
            ),
        }
    )


def assign_split(frame):
    """Purge a full label horizon before both temporal boundaries."""
    t = frame.timestamp
    horizon = pd.Timedelta(hours=HORIZON_HOURS)
    frame = frame.copy()
    frame["split"] = "excluded"
    frame.loc[(t >= "2022-01-01") & (t + horizon < pd.Timestamp("2023-01-01")), "split"] = "train"
    frame.loc[(t >= "2023-01-01") & (t + horizon < pd.Timestamp("2023-07-01")), "split"] = (
        "validation"
    )
    frame.loc[(t >= "2023-07-01") & (t + horizon < pd.Timestamp("2024-01-01")), "split"] = "test"
    return frame


def build_dataset(events, max_gap_hours=MAX_GAP_HOURS):
    canonical = canonicalize(events)
    parts, episodes = [], []
    audit = {
        "canonical_events": len(canonical),
        "ambiguous_timestamps": int(canonical.state.eq(AMBIGUOUS).sum()),
        "max_observation_gap_hours": max_gap_hours,
        "horizon_hours": HORIZON_HOURS,
        "prediction_step_hours": 1,
        "candidate_hours": 0,
        "censored_hours": 0,
        "confirmed_onsets": 0,
        "channels_with_eligible_hours": 0,
    }
    for i, (channel, group) in enumerate(canonical.groupby("channel_id", sort=True)):
        group = group.reset_index(drop=True)
        line = timeline(group, max_gap_hours)
        for idx in np.flatnonzero(line["onset"]):
            episodes.append({"channel_id": channel, "onset": group.timestamp.iloc[idx]})
        start = (group.timestamp.iloc[0] + pd.Timedelta(hours=24)).ceil("h")
        end = min(
            group.timestamp.iloc[-1] + pd.Timedelta(hours=max_gap_hours), pd.Timestamp("2024-01-01")
        )
        if start > end:
            continue
        features = features_at(group, pd.date_range(start, end, freq="h"), max_gap_hours)
        if features.empty:
            continue
        labels = label_at(group, features.timestamp, max_gap_hours)
        audit["candidate_hours"] += len(features)
        audit["censored_hours"] += int(labels[TARGET].eq(-1).sum())
        audit["channels_with_eligible_hours"] += 1
        frame = pd.concat([features.reset_index(drop=True), labels], axis=1)
        parts.append(frame.loc[frame[TARGET].ge(0)].copy())
        if i % 500 == 0:
            print(f"features: {i + 1}/{canonical.channel_id.nunique()} channels", flush=True)
    if not parts:
        raise ValueError("No observed labels under the configured gap rule")
    dataset = assign_split(pd.concat(parts, ignore_index=True))
    dataset = dataset.sort_values(["timestamp", "channel_id"]).reset_index(drop=True)
    episode_frame = pd.DataFrame(episodes, columns=["channel_id", "onset"])
    audit["confirmed_onsets"] = len(episodes)
    audit["labelled_hours"] = len(dataset)
    audit["labelled_channels"] = int(dataset.channel_id.nunique())
    audit["splits"] = {
        split: {
            "rows": len(g),
            "positive_rows": int(g[TARGET].sum()),
            "channels": int(g.channel_id.nunique()),
            "positive_episodes": len(
                g.loc[g[TARGET].eq(1), ["channel_id", "next_onset"]].drop_duplicates()
            ),
            "start": str(g.timestamp.min()),
            "end": str(g.timestamp.max()),
        }
        for split, g in dataset.groupby("split")
    }
    return dataset, episode_frame, audit
