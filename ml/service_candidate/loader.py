"""Load the pinned CBM and score already-computed causal Q2/Q3 feature rows.

This deliberately does not turn uncalibrated scores into operational alarms.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from .features import engineered_input


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(path: Path) -> str:
    """Hash source independent of Git's LF/CRLF checkout convention."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class ResearchRiskModel:
    def __init__(self, directory: Path):
        metadata_path = directory / "model_metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_path = directory / "model.cbm"
        if sha256(model_path) != self.metadata["model_sha256"]:
            raise ValueError("model checksum differs from the pinned metadata")
        if (
            source_sha256(Path(__file__).with_name("features.py"))
            != self.metadata["feature_transform_source_sha256"]
        ):
            raise ValueError("feature transform checksum differs from the pinned metadata")
        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path))
        if self.model.feature_names_ != self.metadata["engineered_feature_names"]:
            raise ValueError("model feature order differs from the contract")
        if self.metadata["production_approved"] or self.metadata["automatic_actions_allowed"]:
            raise ValueError("research package must not claim operational approval")

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return an uncalibrated score or an explicit unavailability status.

        `frame` contains the base features, `admission_status`, and optional
        `channel_id` / `prediction_time` identity. Labels are never accepted.
        """
        forbidden = set(frame.columns) & {
            "target",
            "label_status",
            "target_episode_id",
            "label_available_at",
            "split_status",
        }
        if forbidden:
            raise ValueError(f"future/label fields are not service inputs: {sorted(forbidden)}")
        names = self.metadata["base_feature_names"]
        missing = set(names + ["admission_status"]) - set(frame.columns)
        if missing:
            raise ValueError(f"missing required service inputs: {sorted(missing)}")
        if (
            frame["admission_status"].isna().any()
            or not frame["admission_status"].isin(["eligible", "unknown", "excluded"]).all()
        ):
            raise ValueError("every row needs a valid explicit admission status")
        result = frame[[name for name in ("channel_id", "prediction_time") if name in frame]].copy()
        result["prediction_status"] = np.where(
            frame.admission_status.eq("eligible"), "scored_research", "not_available"
        )
        result["risk_score"] = np.nan
        result["warning"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
        eligible = frame.admission_status.eq("eligible")
        if eligible.any():
            matrix = engineered_input(frame.loc[eligible], names)
            if matrix.columns.tolist() != self.metadata["engineered_feature_names"]:
                raise ValueError("engineered feature order differs from the contract")
            result.loc[eligible, "risk_score"] = self.model.predict_proba(matrix, thread_count=2)[
                :, 1
            ]
        return result
