"""Past-only Q2/Q3 feature transformation from the neighboring experiment.

The algorithm is intentionally frozen for the exported CatBoost model. It
reads only the 121 causal base feature fields, not labels, IDs or future data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineered_input(frame: pd.DataFrame, base_names: list[str]) -> pd.DataFrame:
    if len(base_names) != 121 or len(base_names) != len(set(base_names)):
        raise ValueError("the pinned Q2/Q3 input requires 121 unique features")
    if set(base_names) - set(frame.columns):
        raise ValueError("missing base features")
    output = frame[base_names].copy()
    output["sensor_type"] = output.sensor_type.fillna("<unknown>").astype(str)
    numeric_names = [name for name in base_names if name != "sensor_type"]
    for name in numeric_names:
        output[name] = (
            pd.to_numeric(output[name], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
        )
    extra = {}
    count_names = [
        name
        for name in numeric_names
        if not name.startswith("missing__")
        and ("count" in name or name.startswith("state_transitions_"))
    ]
    for name in count_names:
        extra[f"log1p__{name}"] = np.log1p(output[name].clip(lower=0))
    families = sorted(
        {
            name.rsplit("_", 1)[0]
            for name in count_names
            if name.endswith("_1h") and "distinct" not in name
        }
    )
    for family in families:
        for recent, outer in ((1, 6), (6, 24), (24, 168)):
            short_name, long_name = f"{family}_{recent}h", f"{family}_{outer}h"
            if short_name not in output or long_name not in output:
                continue
            short, long = output[short_name].clip(lower=0), output[long_name].clip(lower=0)
            previous = (long - short).clip(lower=0)
            extra[f"disjoint_log_rate__{family}_{recent}v{outer}h"] = np.log1p(
                short / recent
            ) - np.log1p(previous / (outer - recent))
            extra[f"recent_share__{family}_{recent}v{outer}h"] = short / (long + 1.0)
    for age_name in (
        "last_observation_age_seconds",
        "last_completed_episode_end_age_seconds",
    ):
        age = output[age_name].clip(lower=0) / 3600.0
        extra[f"log1p_hours__{age_name}"] = np.log1p(age)
        extra[f"freshness24__{age_name}"] = np.exp(-age / 24.0)
        extra[f"freshness168__{age_name}"] = np.exp(-age / 168.0)
    freshness = extra["freshness24__last_observation_age_seconds"]
    for family in (
        "event_count",
        "technical_message_count",
        "registered_fault_text_count",
        "normal_message_count",
        "unknown_state_count",
        "state_transitions",
    ):
        name = f"{family}_168h"
        extra[f"fresh_logcount__{name}"] = np.log1p(output[name].clip(lower=0)) * freshness
    for window in (1, 6, 24, 168):
        denominator = output[f"event_count_{window}h"].clip(lower=0) + 1.0
        for family in (
            "technical_message_count",
            "registered_fault_text_count",
            "normal_message_count",
            "environmental_alarm_count",
            "unknown_state_count",
            "state_transitions",
        ):
            extra[f"per_event__{family}_{window}h"] = (
                output[f"{family}_{window}h"].clip(lower=0) / denominator
            )
    output = pd.concat([output, pd.DataFrame(extra, index=frame.index, dtype="float32")], axis=1)
    output[numeric_names + list(extra)] = output[numeric_names + list(extra)].fillna(-1.0)
    return output
