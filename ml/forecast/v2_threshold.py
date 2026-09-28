"""Select a research warning threshold against full-population episode goals."""

from __future__ import annotations

import math
from collections.abc import Sequence


def choose_threshold(curve: Sequence[dict], *, full_positive_episodes: int,
                     precision_goal: float = 0.7,
                     recall_goal: float = 0.5) -> dict:
    """Use validation metrics only; return no selected threshold when goals fail."""
    if (not curve or type(full_positive_episodes) is not int
            or full_positive_episodes <= 0
            or not 0 < precision_goal < 1
            or not 0 < recall_goal < 1):
        raise ValueError("invalid threshold selection population or goals")
    evaluated = []
    for item in curve:
        matched = item["matched_episodes"]
        emitted = item["emitted_warnings"]
        eligible = item["eligible_positive_episodes"]
        threshold = float(item["threshold"])
        if (not all(type(value) is int for value in (matched, emitted, eligible))
                or not 0 <= matched <= eligible <= full_positive_episodes
                or emitted < matched or not math.isfinite(threshold)):
            raise ValueError("invalid validation warning counts or threshold")
        precision = matched / emitted if emitted else 0.0
        conditional_recall = matched / eligible if eligible else 0.0
        if (not math.isfinite(item["episode_precision"])
                or not math.isfinite(item["episode_recall"])
                or abs(item["episode_precision"] - precision) > 1e-12
                or abs(item["episode_recall"] - conditional_recall) > 1e-12):
            raise ValueError("saved validation warning metrics are inconsistent")
        full_recall = matched / full_positive_episodes
        full_f1 = (2 * precision * full_recall / (precision + full_recall)
                   if precision + full_recall else 0.0)
        evaluated.append({
            "threshold": threshold,
            "matched_episodes": matched,
            "emitted_warnings": emitted,
            "episode_precision": precision,
            "conditional_episode_recall": conditional_recall,
            "full_episode_recall": full_recall,
            "full_episode_f1": full_f1,
            "meets_goals": precision > precision_goal and full_recall > recall_goal,
        })
    feasible = [item for item in evaluated if item["meets_goals"]]
    selected = max(feasible, key=lambda item: (
        item["full_episode_recall"], item["episode_precision"],
        -item["emitted_warnings"], item["threshold"],
    )) if feasible else None
    diagnostic = max(evaluated, key=lambda item: (
        item["full_episode_f1"], item["full_episode_recall"],
        item["episode_precision"], -item["emitted_warnings"], item["threshold"],
    ))
    return {
        "requirements_feasible_on_checked_grid": selected is not None,
        "selected": selected,
        "diagnostic_best_full_f1": diagnostic,
        "thresholds_examined": len(evaluated),
    }
