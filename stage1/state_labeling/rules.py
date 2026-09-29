"""R1/B1 semantics: interpret explicit messages independently from alarm.

Dictionary matches are candidates: the supplied CSV has no channel-to-state-set
key. Only the exact project rule for a known channel type selects the first
registered-state target. Episode and training-label rules live in separate modules.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path


RULESET_VERSION = "registered-state-r1-operational-v2"
STATE_DICTIONARY_SHA256 = "b968f3652ed33fd6e0d2dccee334275436dada67f4549763cfb4b2072108b5cc"
TARGET_DEFINITION = {
    "version": RULESET_VERSION,
    "status": "accepted_operational_archive_assumption",
    "target_kind": "registered_neispraven_onset",
    "message_text": "Неисправен",
    "horizon_hours": 24,
    "included_years": (2019, 2020, 2022, 2023, 2024, 2025, 2026),
    "archive_segments": (
        ("2019-01-01T00:00:00", "2021-01-01T00:00:00"),
        ("2022-01-01T00:00:00", "2026-07-01T00:00:00"),
    ),
    "source_policy": "seven_full_ext_journal_archives_only",
    "coverage_assumption": "archive_complete_assumed_not_channel_continuity_verified",
    "recent_normal_hours": 168,
    "recovery_rule": "one_exact_norma_same_channel_no_same_second_conflict",
    "unknown_type_policy": "unknown_not_positive",
    "future_window": "(prediction_time, prediction_time + 24h]",
    "meaning": "new registered message episode, not verified physical failure",
    "source": "project_rule",
}

KNOWN_SENSOR_TYPES = frozenset(
    {
        "9-секционный люк",
        "Газовый датчик",
        "Датчик движения",
        "Датчик дыма",
        "Датчик затопления",
        "Датчик температуры",
        "ИБП",
        "КД АВ",
        "КД Дверь",
        "КД Люк",
        "Переключатель",
        "Ручной извещатель",
        "Состояние УИР-Р",
        "Состояние вентилятора",
        "Состояние насоса",
        "Состояние охраны",
        "Состояние фазы",
        "Стекло",
        "Тепловой датчик",
    }
)
CATEGORIES = frozenset(
    {"normal", "operational", "environmental_alarm", "technical_fault", "unknown"}
)
SOURCES = frozenset({"dictionary_candidate", "project_rule", "unresolved"})


@dataclass(frozen=True, slots=True)
class MessageInterpretation:
    sensor_type: str | None
    value_state: str | None
    observed_alarm: bool | None
    category: str
    source: str
    reason: str
    status: str
    target_message_candidate: bool
    ruleset_version: str = RULESET_VERSION

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES or self.source not in SOURCES:
            raise ValueError("invalid semantic category or source")
        if self.target_message_candidate and (
            self.category != "technical_fault" or self.source != "project_rule"
        ):
            raise ValueError("target event requires an explicit technical-fault project rule")


# Every unique (type, text) in the supplied CSV is deliberately assigned below.
# A candidate category describes the text, not a proved historic state-set join.
# Suspicious type/text combinations and contradictory definitions remain unknown.
_GROUPS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("КД Дверь", "normal", "explicit_normal_text", ("Норма",)),
    ("КД Дверь", "operational", "contact_or_switch_state", ("Выключен", "Замкнут")),
    ("КД Дверь", "unknown", "disabled_or_undefined_ambiguous", ("Отключено устройство", "Неопределен")),
    ("КД Люк", "normal", "explicit_normal_text", ("Норма",)),
    ("КД Люк", "operational", "contact_or_switch_state", ("Не замкнут", "Включен", "Выключен")),
    ("КД АВ", "unknown", "type_state_association_requires_review", ("Дыма нет", "Обнаружен дым", "Движения нет", "Обнаружено движение", "Движение вверх", "Движение вниз", "Движение влево", "Движение вправо")),
    ("Датчик температуры", "normal", "explicit_normal_text", ("Норма",)),
    ("Датчик температуры", "unknown", "type_state_association_requires_review", ("Обнаружен газ",)),
    ("Датчик движения", "operational", "switch_state", ("Выключен", "Включен")),
    ("Датчик движения", "unknown", "power_or_type_association_requires_review", ("Затоплен", "Обесточен", "Работают все насосы в АНС")),
    ("Датчик дыма", "normal", "explicit_normal_text", ("Рычаг норма",)),
    ("Датчик дыма", "operational", "lever_or_power_state", ("Рычаг сдернут", "Рычаг сдернут влево", "Рычаг сдернут вправо", "Оба рычага сдернуты", "Есть питание")),
    ("Датчик дыма", "unknown", "power_loss_not_proven_device_failure", ("Обесточен",)),
    ("Газовый датчик", "normal", "explicit_normal_text", ("Норма",)),
    ("Газовый датчик", "unknown", "type_state_association_requires_review", ("Разговор", "Вызов", "Температура выше 35ºC1", "В норме от +3 до +351", "Температура ниже 3ºC", "Не определено", "В норме от +3 до +40", "Температура выше 40ºC", "В норме от +3 до +27ºC", "Температура выше 27ºC")),
    ("Газовый датчик", "unknown", "conflicting_dictionary_alarm", ("Температура ниже 3ºC1",)),
    ("Состояние вентилятора", "operational", "power_source_state", ("Питание от сети", "Питание от батарей")),
    ("Состояние вентилятора", "technical_fault", "explicit_battery_fault_text", ("Батарея неисправна", "Батарея разряжена")),
    ("Состояние вентилятора", "normal", "explicit_aggregate_health_text", ("Устройства на объекте исправны",)),
    ("Состояние вентилятора", "technical_fault", "explicit_aggregate_fault_text", ("Много неисправных устройств",)),
    ("Состояние вентилятора", "unknown", "type_state_association_requires_review", ("Включены все насосы АНС",)),
    ("Состояние насоса", "operational", "security_mode_state", ("На охране", "Снято с охраны")),
    ("Состояние фазы", "unknown", "type_state_association_requires_review", ("Работают все насосы АНС",)),
)


def _candidate_rules() -> dict[tuple[str, str], tuple[str, str]]:
    rules: dict[tuple[str, str], tuple[str, str]] = {}
    for sensor_type, category, reason, states in _GROUPS:
        for state in states:
            key = (sensor_type, state)
            if key in rules:
                raise AssertionError(f"duplicate semantic rule: {key}")
            rules[key] = (category, reason)
    return rules


DICTIONARY_CANDIDATES = _candidate_rules()
DICTIONARY_PAIRS = frozenset(DICTIONARY_CANDIDATES) | {("КД Дверь", "Неисправен")}


def classify_message(
    sensor_type: str | None, value_state: str | None, alarm: bool | None
) -> MessageInterpretation:
    """Return a versioned, conservative interpretation of one normalized event.

    Matching is exact. It never trims, changes case, repairs source types, or
    uses the observed alarm to pick a category or dictionary candidate.
    """
    if alarm is not None and not isinstance(alarm, bool):
        raise TypeError("alarm must be bool or None")

    def result(category: str, source: str, reason: str, status: str, target=False):
        return MessageInterpretation(
            sensor_type, value_state, alarm, category, source, reason, status, target
        )

    if sensor_type not in KNOWN_SENSOR_TYPES:
        return result("unknown", "unresolved", "unverified_sensor_type", "unknown_type")
    if value_state is None:
        return result("unknown", "unresolved", "no_text_state", "not_state_message")
    if value_state == "Неисправен":
        return result(
            "technical_fault", "project_rule", "exact_registered_fault_text",
            "project_defined", True,
        )
    if value_state == "Норма":
        return result("normal", "project_rule", "exact_normal_text", "project_defined")
    if value_state == "Неопределен":
        return result("unknown", "project_rule", "explicit_undefined_text", "unknown_state")
    if (sensor_type, value_state) in {
        ("Датчик дыма", "Обнаружен дым"),
        ("Датчик движения", "Обнаружено движение"),
    }:
        return result(
            "environmental_alarm", "project_rule", "observed_environmental_event",
            "project_defined",
        )
    if (sensor_type, value_state) in {
        ("Датчик дыма", "Дыма нет"),
        ("Датчик движения", "Движения нет"),
    }:
        return result("normal", "project_rule", "observed_clear_state", "project_defined")
    if (sensor_type, value_state) == ("КД Дверь", "Не замкнут"):
        return result("operational", "project_rule", "observed_contact_state", "project_defined")
    candidate = DICTIONARY_CANDIDATES.get((sensor_type, value_state))
    if candidate is not None:
        category, reason = candidate
        if category == "unknown":
            return result(category, "unresolved", reason, "requires_review")
        return result(category, "dictionary_candidate", reason, "candidate_only")
    return result("unknown", "unresolved", "unmapped_state", "unknown_state")


def in_scope_year(year: int) -> bool:
    """The omitted year is an intentional study boundary."""
    return year in TARGET_DEFINITION["included_years"]


def review_dictionary(path: Path) -> dict:
    """Check complete semantic coverage of the *supplied* CSV without changing it."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"тип_датчика", "ид_набор_состояний", "название_состояния", "тревожное"}
        if reader.fieldnames is None or (
            set(reader.fieldnames) != required or len(reader.fieldnames) != len(required)
        ):
            raise ValueError("unexpected state dictionary schema")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("malformed state dictionary row")
    full = [
        (r["тип_датчика"], r["ид_набор_состояний"], r["название_состояния"], r["тревожное"])
        for r in rows
    ]
    pairs = {(r[0], r[2]) for r in full}
    alarm_by_key: dict[tuple[str, str, str], set[str]] = {}
    sets_by_pair: dict[tuple[str, str], set[str]] = {}
    for sensor_type, set_id, state, alarm in full:
        if alarm not in {"true", "false"}:
            raise ValueError("unexpected dictionary alarm value")
        alarm_by_key.setdefault((sensor_type, set_id, state), set()).add(alarm)
        sets_by_pair.setdefault((sensor_type, state), set()).add(set_id)
    contradictions = sorted(key for key, values in alarm_by_key.items() if len(values) > 1)
    multiple_sets = sorted(
        (sensor_type, state, sorted(set_ids))
        for (sensor_type, state), set_ids in sets_by_pair.items()
        if len(set_ids) > 1
    )
    covered = pairs & DICTIONARY_PAIRS
    missing = pairs - DICTIONARY_PAIRS
    extra = DICTIONARY_PAIRS - pairs
    review_rows = []
    for sensor_type, set_id, state, alarm in dict.fromkeys(full):
        interpretation = classify_message(sensor_type, state, alarm == "true")
        review_rows.append(
            {
                "sensor_type": sensor_type,
                "state_set_id_candidate": set_id,
                "value_state": state,
                "dictionary_alarm": alarm == "true",
                "category": interpretation.category,
                "source": interpretation.source,
                "reason": interpretation.reason,
                "target_message_candidate": interpretation.target_message_candidate,
            }
        )
    return {
        "ruleset_version": RULESET_VERSION,
        "target_status": TARGET_DEFINITION["status"],
        "dictionary_sha256": digest,
        "expected_dictionary_sha256": STATE_DICTIONARY_SHA256,
        "dictionary_hash_matches": digest == STATE_DICTIONARY_SHA256,
        "rows": len(full),
        "unique_rows": len(set(full)),
        "unique_type_state_pairs": len(pairs),
        "covered_type_state_pairs": len(covered),
        "unreviewed_pairs": sorted(missing),
        "rules_absent_from_dictionary": sorted(extra),
        "contradictory_type_set_states": contradictions,
        "multiple_candidate_sets": multiple_sets,
        "review_rows": review_rows,
        "dictionary_review_complete": not missing and not extra and digest == STATE_DICTIONARY_SHA256,
        "ready_for_joint_review": not missing and not extra and digest == STATE_DICTIONARY_SHA256,
    }
