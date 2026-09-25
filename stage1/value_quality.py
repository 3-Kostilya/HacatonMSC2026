"""Versioned QA interpretation of raw values, separate from forecast labels.

These rules mark candidate data-quality and environmental signals. They never
declare a physical device failure or create a positive forecast target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


QA_VALUE_RULESET_VERSION = "qa-values-v1"
EPOCH_VALUE_ARTIFACTS = frozenset(
    {
        "01.01.1970 03:00:00",
        "01.01.1970 03:00:01",
    }
)
TEMPERATURE_SERVICE_CODE_CANDIDATES = frozenset(
    {
        -3276.0,
        -255.0,
        -127.0,
        -100.0,
        255.0,
        999.0,
    }
)
GAS_ALARM_THRESHOLD_PERCENT_VOLUME = 1.0
GAS_PHYSICAL_MAX_PERCENT_VOLUME = 100.0


@dataclass(frozen=True, slots=True)
class ValueQuality:
    category: str
    numeric_measurement_usable: bool
    state_transition_usable: bool
    environmental_alarm_level_candidate: bool = False
    technical_code_candidate: bool = False
    ruleset_version: str = QA_VALUE_RULESET_VERSION


def assess_value(
    sensor_type: str | None,
    raw_value: str,
    numeric_value: float | None,
) -> ValueQuality:
    """Classify one observation with exact, documented type-specific guards.

    Negative methane readings are kept in raw numeric dynamics. Their physical
    meaning is unclear, and excluding around a million small negative readings
    would erase a possible calibration signal. They are flagged for audit.
    """

    if numeric_value is not None and not math.isfinite(numeric_value):
        raise ValueError("numeric_value must be finite")
    if numeric_value is None:
        if raw_value.strip() in EPOCH_VALUE_ARTIFACTS:
            return ValueQuality("epoch_value_artifact", False, False)
        return ValueQuality("ordinary_text", False, True)
    if sensor_type == "Датчик температуры" and numeric_value in TEMPERATURE_SERVICE_CODE_CANDIDATES:
        return ValueQuality(
            "temperature_service_code_candidate", False, False, technical_code_candidate=True
        )
    if sensor_type == "Газовый датчик":
        if numeric_value < 0:
            return ValueQuality("gas_negative_reading", True, False)
        if numeric_value > GAS_PHYSICAL_MAX_PERCENT_VOLUME:
            return ValueQuality(
                "gas_above_physical_percent", False, False, technical_code_candidate=True
            )
        if numeric_value >= GAS_ALARM_THRESHOLD_PERCENT_VOLUME:
            return ValueQuality(
                "gas_alarm_level_candidate", True, False, environmental_alarm_level_candidate=True
            )
    return ValueQuality("ordinary_numeric", True, False)
