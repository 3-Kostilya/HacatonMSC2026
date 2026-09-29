"""Typed access to the validated detector applicability matrix."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "stage1" / "config" / "type_detector_matrix.yaml"
VALID_MODES = frozenset({"numeric", "discrete", "context"})
VALID_STATUSES = frozenset({"primary", "supplemental", "conditional", "not_applicable"})


@dataclass(frozen=True, slots=True)
class TypePolicy:
    sensor_type: str
    family: str
    expected_channels: int
    modes: dict[str, str]
    features: tuple[str, ...]
    mandatory_checks: tuple[str, ...]
    unknown_if: tuple[str, ...]
    inapplicability_rule: str
    interpretation_guard: str

    def supports(self, mode: str) -> bool:
        if mode not in VALID_MODES:
            raise ValueError(f"unknown processing mode: {mode}")
        return self.modes[mode] != "not_applicable"

    def requires_confirmation(self, mode: str) -> bool:
        if mode not in VALID_MODES:
            raise ValueError(f"unknown processing mode: {mode}")
        return self.modes[mode] == "conditional"


class TypeRegistry:
    def __init__(self, policies: Iterable[TypePolicy], ruleset_version: str):
        self.ruleset_version = ruleset_version
        self._policies = {policy.sensor_type: policy for policy in policies}
        if len(self._policies) != 19:
            raise ValueError(f"registry must contain 19 unique types, got {len(self._policies)}")

    def __len__(self) -> int:
        return len(self._policies)

    def __iter__(self):
        return iter(self._policies.values())

    def get(self, sensor_type: str) -> TypePolicy:
        try:
            return self._policies[sensor_type]
        except KeyError as exc:
            raise KeyError(
                f"sensor type is absent from the validated registry: {sensor_type}"
            ) from exc

    def by_mode(self, mode: str, include_conditional: bool = True) -> tuple[TypePolicy, ...]:
        if mode not in VALID_MODES:
            raise ValueError(f"unknown processing mode: {mode}")
        allowed = {"primary", "supplemental"}
        if include_conditional:
            allowed.add("conditional")
        return tuple(policy for policy in self if policy.modes[mode] in allowed)


def load_registry(path: Path = DEFAULT_MATRIX) -> TypeRegistry:
    data = json.loads(path.read_text(encoding="utf-8"))
    policies = []
    for raw in data.get("types", []):
        modes = raw.get("modes", {})
        if set(modes) != VALID_MODES:
            raise ValueError(f"{raw.get('sensor_type')}: modes must be {sorted(VALID_MODES)}")
        invalid = set(modes.values()) - VALID_STATUSES
        if invalid:
            raise ValueError(f"{raw.get('sensor_type')}: invalid statuses {sorted(invalid)}")
        policies.append(
            TypePolicy(
                sensor_type=raw["sensor_type"],
                family=raw["family"],
                expected_channels=raw["expected_channels"],
                modes=dict(modes),
                features=tuple(raw["features"]),
                mandatory_checks=tuple(raw["mandatory_checks"]),
                unknown_if=tuple(raw["unknown_if"]),
                inapplicability_rule=raw["inapplicability_rule"],
                interpretation_guard=raw["interpretation_guard"],
            )
        )
    return TypeRegistry(policies, ruleset_version=f"stage1-matrix-v{data['schema_version']}")
