"""Frozen component scores for the research-only Round 7 candidate.

This module deliberately stops before warning emission. The published result
also depends on Q2/Q3 causal admission, chronological registered-state groups,
recovery reset and a 24-hour cooldown. Scores alone do not reproduce it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd

from ml.service_candidate.features import engineered_input


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = Path(__file__).resolve().parent
FORBIDDEN_FIELDS = frozenset(
    {"target", "label_status", "target_episode_id", "label_available_at", "split_status"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(path: Path) -> str:
    """Keep the pinned JSON hash stable across Git LF/CRLF checkouts."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _linear_transform(frame: pd.DataFrame, names: list[str], limits: dict[str, float]) -> pd.DataFrame:
    """Exact `log_episode_sqrt` preprocessing from the frozen experiment."""
    output = frame[names].copy()
    output["sensor_type"] = output["sensor_type"].fillna("<unknown>")
    for name in names:
        if name == "sensor_type" or name.startswith("missing__"):
            continue
        values = pd.to_numeric(output[name], errors="coerce").astype("float32")
        if name in limits:
            values = values.clip(upper=limits[name])
        output[name] = np.log1p(values.clip(lower=0))
    for kind in (
        "event_count",
        "alarm_count",
        "state_transitions",
        "registered_fault_text_count",
        "normal_message_count",
    ):
        for short, long in ((1, 6), (6, 24), (24, 168)):
            recent = frame[f"{kind}_{short}h"].astype("float32")
            cumulative = frame[f"{kind}_{long}h"].astype("float32")
            old_rate = (cumulative - recent).clip(lower=0) / (long - short)
            output[f"rate_change__{kind}_{short}_{long}"] = (
                np.log1p(recent / short) - np.log1p(old_rate)
            )
    return output


def _specialist_matrix(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    output = frame[names].copy()
    output["sensor_type"] = output["sensor_type"].astype(str)
    for name in names:
        if name != "sensor_type":
            output[name] = (
                pd.to_numeric(output[name], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(-1)
                .astype("float32")
            )
    return output


class Round7ResearchModels:
    """Hash-pinned components; never labels, alarms or operational approval."""

    def __init__(self, bundle: Path = DEFAULT_BUNDLE):
        bundle = Path(bundle)
        self.manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        self.specialist_selection = json.loads(
            (bundle / "frozen_specialist_selection.json").read_text(encoding="utf-8")
        )
        if (
            self.manifest["schema_version"] != "registered-journal-round7-research-v1"
            or self.manifest["production_approved"]
            or self.manifest["automatic_actions_allowed"]
        ):
            raise ValueError("Round 7 bundle is research-only")
        files = {
            "linear_model_sha256": bundle / "weights" / "linear.joblib",
            "tree_model_sha256": bundle / "weights" / "tree.cbm",
            "specialist_model_sha256": bundle / "weights" / "specialist.cbm",
        }
        for key, path in files.items():
            if _sha256(path) != self.manifest[key]:
                raise ValueError(f"Round 7 artifact checksum differs: {path.name}")
        if _text_sha256(bundle / "frozen_specialist_selection.json") != self.manifest[
            "specialist_selection_sha256"
        ]:
            raise ValueError("specialist selection checksum differs")
        if self.specialist_selection["model_sha256"] != self.manifest["specialist_model_sha256"]:
            raise ValueError("specialist selection belongs to another model")
        if self.specialist_selection["threshold"] != self.manifest["specialist_threshold"]:
            raise ValueError("specialist threshold differs from the frozen selection")

        contract = json.loads(
            (ROOT / "backend" / "data" / "model" / "current" / "model_metadata.json")
            .read_text(encoding="utf-8")
        )
        self.base_names = contract["base_feature_names"]
        self.linear_names = [
            name for name in self.base_names
            if name != self.manifest["linear_feature_omitted_from_121"]
        ]
        if len(self.base_names) != 121 or len(self.linear_names) != 120:
            raise ValueError("Q2/Q3 frozen base feature contract differs")

        # joblib is a pickle format. Only the shipped, checksum-pinned file is loaded.
        self.linear = joblib.load(files["linear_model_sha256"])
        self.tree = CatBoostClassifier()
        self.tree.load_model(str(files["tree_model_sha256"]))
        self.specialist = CatBoostClassifier()
        self.specialist.load_model(str(files["specialist_model_sha256"]))
        if self.tree.feature_names_ != contract["engineered_feature_names"]:
            raise ValueError("tree feature order differs from the frozen contract")
        if self.specialist.feature_names_ != self.specialist_selection["feature_names"]:
            raise ValueError("specialist feature order differs from the frozen selection")

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score admitted causal rows; return static gates, not emitted warnings."""
        forbidden = FORBIDDEN_FIELDS.intersection(frame.columns)
        if forbidden:
            raise ValueError(f"future/label fields are not model inputs: {sorted(forbidden)}")
        required = set(self.base_names) | {
            "channel_id", "prediction_time", "admission_status", "last_explicit_normal_at"
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"missing Round 7 inputs: {sorted(missing)}")
        if frame.duplicated(["channel_id", "prediction_time"]).any():
            raise ValueError("duplicate channel-hour keys")
        if frame["admission_status"].isna().any() or not frame["admission_status"].isin(
            ["eligible", "unknown", "excluded"]
        ).all():
            raise ValueError("explicit Q2/Q3 admission status is required")

        result = frame[["channel_id", "prediction_time"]].copy()
        result["prediction_status"] = np.where(
            frame["admission_status"].eq("eligible"), "scored_research", "not_available"
        )
        for name in ("score_linear", "score_tree", "score_specialist"):
            result[name] = np.nan
        for name in ("passes_common_gates", "passes_standard_gates"):
            result[name] = pd.Series(pd.NA, index=frame.index, dtype="boolean")

        eligible = frame["admission_status"].eq("eligible")
        if not eligible.any():
            return result
        admitted = frame.loc[eligible].copy()
        at = pd.to_datetime(admitted["prediction_time"], errors="raise")
        normal = pd.to_datetime(admitted["last_explicit_normal_at"], errors="raise")
        if at.dt.tz is not None or normal.dt.tz is not None:
            raise ValueError("Round 7 requires naive local journal timestamps")
        if (at.dt.minute.ne(0) | at.dt.second.ne(0) | at.dt.microsecond.ne(0)).any():
            raise ValueError("Round 7 decisions require complete prediction hours")
        if normal.isna().any() or (normal > at).any():
            raise ValueError("eligible row needs a past explicit Norma")
        normal_age = (at - normal).dt.total_seconds() / 3600

        linear = self.linear.predict_proba(
            _linear_transform(admitted, self.linear_names, self.manifest["linear_limits"])
        )[:, 1]
        tree = self.tree.predict_proba(
            engineered_input(admitted, self.base_names), thread_count=2
        )[:, 1]
        result.loc[eligible, "score_linear"] = linear
        result.loc[eligible, "score_tree"] = tree

        sensor_type = admitted["sensor_type"].astype(str)
        threshold = np.where(
            sensor_type.eq("Состояние фазы"),
            self.manifest["phase_linear_threshold"],
            self.manifest["linear_threshold"],
        )
        common = (
            (linear >= threshold)
            & ((tree >= self.manifest["tree_veto_threshold"])
               | sensor_type.eq(self.manifest["tree_veto_exempt_sensor_type"]).to_numpy())
            & (~sensor_type.eq("Датчик дыма").to_numpy()
               | (normal_age.to_numpy() <= self.manifest["smoke_max_normal_age_hours"]))
        )
        standard = common.copy()
        smoke = sensor_type.eq("Датчик дыма").to_numpy()
        unknown_count = pd.to_numeric(
            admitted["unknown_state_count_168h"], errors="coerce"
        ).to_numpy()
        if np.isnan(unknown_count[smoke]).any():
            raise ValueError("smoke veto needs causal unknown-state count")
        standard &= ~smoke | (
            unknown_count <= self.manifest["smoke_max_unknown_state_count_168h"]
        )

        specialist_kind = sensor_type.isin(["Датчик дыма", "Состояние фазы"])
        if specialist_kind.any():
            specialist_rows = admitted.loc[specialist_kind].copy()
            specialist_rows["score_linear"] = linear[specialist_kind.to_numpy()]
            specialist_rows["score_tree"] = tree[specialist_kind.to_numpy()]
            specialist_rows["normal_age_hours"] = normal_age.loc[specialist_kind].astype(
                "float32"
            )
            scores = self.specialist.predict_proba(
                _specialist_matrix(
                    specialist_rows, self.specialist_selection["feature_names"]
                ),
                thread_count=2,
            )[:, 1]
            result.loc[specialist_rows.index, "score_specialist"] = scores
            standard[specialist_kind.to_numpy()] &= (
                scores >= self.manifest["specialist_threshold"]
            )
        result.loc[eligible, "passes_common_gates"] = common
        result.loc[eligible, "passes_standard_gates"] = standard
        return result
