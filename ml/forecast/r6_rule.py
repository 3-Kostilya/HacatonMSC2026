"""Frozen R6 score for conditionally admitted registered-journal forecasts.

The caller must supply admission status. Unknown or excluded hours never get
a zero-risk score; they remain unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RULE_VERSION = "r4-b-past-state-counts-v1"
TERMS = {
    "registered_fault_text_count_24h": 2.0,
    "registered_fault_text_count_168h": 0.5,
    "completed_episode_count_168h": 1.0,
    "technical_message_count_24h": 0.1,
}


def predict_rule(frame: pd.DataFrame, *, eligibility_status: pd.Series | str,
                 threshold: float) -> pd.DataFrame:
    """Score eligible feature rows with the accepted R4 formula.

    Missing counts keep R4's established zero-fill behavior. Missing *admission*
    is never zero-filled: a caller must explicitly mark every eligible row.
    """

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    missing = set(TERMS) - set(frame.columns)
    if missing:
        raise ValueError(f"required rule features absent: {sorted(missing)}")
    if isinstance(eligibility_status, str):
        statuses = pd.Series(eligibility_status, index=frame.index)
    else:
        statuses = eligibility_status.reindex(frame.index)
    if statuses.isna().any():
        raise ValueError("eligibility status must be explicit for every row")
    ready = statuses.eq("eligible")
    result = pd.DataFrame(index=frame.index)
    result["prediction_status"] = statuses.astype(str)
    result["rule_score"] = np.nan
    result["alert"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    if ready.any():
        score = np.zeros(int(ready.sum()), dtype=np.float64)
        for name, weight in TERMS.items():
            values = pd.to_numeric(frame.loc[ready, name], errors="coerce")
            score += weight * values.fillna(0).to_numpy(dtype=np.float64)
        result.loc[ready, "rule_score"] = score
        result.loc[ready, "alert"] = score >= threshold
    return result
