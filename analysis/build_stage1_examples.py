"""Build a traceable, non-expert review of every channel in the Day-3 sample.

The report deliberately includes every catalog record (38 at the time of the
review).  It is a technical inspection of the pipeline, not ground truth about
the physical condition of equipment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "output" / "stage1" / "baseline_catalog.json"
DEFAULT_EVENTS = ROOT / "output" / "stage1" / "normalized_sample.parquet"
DEFAULT_OUTPUT = ROOT / "output" / "stage1" / "real_examples.json"
DEFAULT_REPORT = ROOT / "docs" / "stage1-real-examples.md"

# The seven reporting groups are copied from the agreed stage-one plan.  The
# detector's finer-grained sensor_group is retained separately in every record.
REPORT_GROUPS: dict[str, tuple[str, ...]] = {
    "Числовые измерения среды": ("Датчик температуры", "Газовый датчик"),
    "Пожарные извещатели": ("Датчик дыма", "Тепловой датчик"),
    "Охранные и контактные каналы": (
        "Датчик движения",
        "КД АВ",
        "КД Дверь",
        "КД Люк",
        "9-секционный люк",
        "Стекло",
    ),
    "Контроль затопления": ("Датчик затопления",),
    "Состояние оборудования": ("Состояние насоса", "Состояние вентилятора"),
    "Питание и состояние аппаратуры": ("ИБП", "Состояние фазы", "Состояние УИР-Р"),
    "Управление, ручные сигналы и режим охраны": (
        "Переключатель",
        "Ручной извещатель",
        "Состояние охраны",
    ),
}
TYPE_TO_GROUP = {
    sensor_type: group
    for group, sensor_types in REPORT_GROUPS.items()
    for sensor_type in sensor_types
}

INTERPRETATION_LIMITS = {
    "candidate": (
        "Технический кандидат по правилу baseline; без экспертной разметки это не "
        "подтверждение физической неисправности."
    ),
    "no_candidate": (
        "Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности."
    ),
    "unknown": (
        "Решение намеренно не принято из-за качества или недостатка истории; состояние "
        "оборудования неизвестно."
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp(value: Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().isoformat(sep=" ")
    return value


def _row_reference(row: dict[str, Any], *, start: datetime, confirmed: datetime) -> dict[str, Any]:
    timestamp = _timestamp(row["timestamp"])
    if timestamp < start:
        relation = "baseline_context"
    elif timestamp <= confirmed:
        relation = "episode_interval"
    else:
        relation = "followup_context"
    return {
        "source": row["source"],
        "source_row": row["source_row"],
        "event_id": row["event_id"],
        "timestamp": _json_value(row["timestamp"]),
        "raw_value": row["raw_value"],
        "numeric_value": row["numeric_value"],
        "alarm": row["alarm"],
        "quality_flags": list(row["quality_flags"] or ()),
        "relation_to_episode": relation,
    }


def _trace_rows(rows: list[dict[str, Any]], detector: dict[str, Any]) -> list[dict[str, Any]]:
    """Select up to six raw row references spanning start through confirmation."""
    ordered = sorted(rows, key=lambda row: (_timestamp(row["timestamp"]), row["source_row"]))
    start = datetime.fromisoformat(detector["start_at"])
    confirmed = datetime.fromisoformat(detector["confirmed_at"])
    in_interval = [row for row in ordered if start <= _timestamp(row["timestamp"]) <= confirmed]
    if not in_interval:
        in_interval = sorted(
            ordered,
            key=lambda row: abs((_timestamp(row["timestamp"]) - start).total_seconds()),
        )[:1]
    indices = {ordered.index(row) for row in in_interval}
    first = min(indices)
    last = max(indices)
    if first:
        indices.add(first - 1)
    if last + 1 < len(ordered):
        indices.add(last + 1)
    selected = [ordered[index] for index in sorted(indices)]
    if len(selected) > 6:
        selected = selected[:3] + selected[-3:]
    return [_row_reference(row, start=start, confirmed=confirmed) for row in selected]


def _channel_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: _timestamp(row["timestamp"]))
    values = Counter(str(row["raw_value"]) for row in ordered)
    numeric = [float(row["numeric_value"]) for row in ordered if row["numeric_value"] is not None]
    flags = Counter(flag for row in ordered for flag in (row["quality_flags"] or ()))
    summary: dict[str, Any] = {
        "event_count": len(ordered),
        "first_at": _json_value(ordered[0]["timestamp"]),
        "last_at": _json_value(ordered[-1]["timestamp"]),
        "alarm_true_count": sum(bool(row["alarm"]) for row in ordered),
        "distinct_raw_values": len(values),
        "most_common_raw_values": [
            {"value": value, "count": count} for value, count in values.most_common(5)
        ],
        "quality_flags": dict(sorted(flags.items())),
    }
    if numeric:
        summary["numeric"] = {
            "count": len(numeric),
            "min": min(numeric),
            "max": max(numeric),
        }
    return summary


def build_examples(catalog_path: Path, events_path: Path) -> dict[str, Any]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    event_rows = [
        row
        for row in pq.read_table(events_path).to_pylist()
        if row["disposition"] == "accepted" and row["channel_id"]
    ]
    by_channel: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        by_channel[(str(row["sensor_type"]), str(row["channel_id"]))].append(row)

    examples = []
    for number, record in enumerate(catalog["records"], start=1):
        detector = record["detector"]
        key = (record["sensor_type"], record["channel_id"])
        rows = by_channel.get(key, [])
        if not rows:
            raise ValueError(f"catalog channel has no accepted rows: {key}")
        decision = detector["decision"]
        examples.append(
            {
                "example_id": f"EX-{number:02d}",
                "channel_id": record["channel_id"],
                "sensor_type": record["sensor_type"],
                "report_group": TYPE_TO_GROUP[record["sensor_type"]],
                "detector_group": detector["sensor_group"],
                "processing_mode": record["processing_mode"],
                "decision": decision,
                "episode": detector,
                "episodes": record.get("detector_results", [detector]),
                "evidence": detector["evidence"],
                "observation_quality": {
                    "detector_flags": detector["observation_quality"],
                    "status": record["observability"]["status"],
                    "reasons": record["observability"]["reasons"],
                    "history_event_count": record["observability"]["history_event_count"],
                    "history_sufficient": record["observability"]["history_sufficient"],
                },
                "available_context": {
                    "object_id": detector["object_id"],
                    "context_status": record["context_status"],
                    "context_reason": record["context_reason"],
                    "channel_summary": _channel_summary(rows),
                },
                "source_rows": _trace_rows(rows, detector),
                "interpretation_limit": INTERPRETATION_LIMITS[decision],
                "trace": [
                    "source+source_row",
                    "accepted normalized event with raw_value preserved",
                    f"{record['processing_mode']} detector / {detector['anomaly_type']}",
                    f"Episode {detector['episode_id']} / {decision}",
                ],
            }
        )

    decisions = Counter(example["decision"] for example in examples)
    types = {example["sensor_type"] for example in examples}
    groups = {example["report_group"] for example in examples}
    unknown_reasons = Counter(
        reason
        for example in examples
        if example["decision"] == "unknown"
        for reason in example["observation_quality"]["detector_flags"]
    )
    episode_decisions = Counter(
        episode["decision"] for example in examples for episode in example["episodes"]
    )
    validation = {
        "example_count_between_30_and_50": 30 <= len(examples) <= 50,
        "all_19_types_present": len(types) == 19,
        "all_7_reporting_groups_present": len(groups) == 7,
        "all_three_decisions_present": set(decisions)
        == {
            "candidate",
            "no_candidate",
            "unknown",
        },
        "every_example_has_source_row": all(example["source_rows"] for example in examples),
        "every_example_has_episode_row": all(
            any(row["relation_to_episode"] == "episode_interval" for row in example["source_rows"])
            for example in examples
        ),
        "every_unknown_has_reason": all(
            example["observation_quality"]["detector_flags"]
            for example in examples
            if example["decision"] == "unknown"
        ),
        "all_catalog_episodes_preserved": sum(len(example["episodes"]) for example in examples)
        == catalog.get("episodes", len(examples)),
    }
    if not all(validation.values()):
        raise ValueError(f"review validation failed: {validation}")

    return {
        "scope": "technical pipeline review; not expert labeling or proof of a fault",
        "ruleset": catalog["ruleset"],
        "inputs": {
            "catalog": str(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
            "events": str(events_path),
            "events_sha256": _sha256(events_path),
            "sample_only": catalog["sample_only"],
            "cadence_confirmed": catalog["cadence_confirmed"],
        },
        "summary": {
            "examples": len(examples),
            "types": len(types),
            "reporting_groups": len(groups),
            "decisions": dict(sorted(decisions.items())),
            "episodes": sum(episode_decisions.values()),
            "episode_decisions": dict(sorted(episode_decisions.items())),
            "unknown_detector_reasons": dict(sorted(unknown_reasons.items())),
        },
        "validation": validation,
        "examples": examples,
    }


def _short_rows(example: dict[str, Any]) -> str:
    return "; ".join(f"`{row['source']}:{row['source_row']}`" for row in example["source_rows"][:3])


def _reason(example: dict[str, Any]) -> str:
    evidence = example["evidence"]
    if evidence:
        return "; ".join(evidence)
    return "; ".join(example["observation_quality"]["detector_flags"])


def _quality(example: dict[str, Any]) -> str:
    quality = example["observation_quality"]
    details = quality["detector_flags"] or quality["reasons"]
    return f"{quality['status']}: " + "; ".join(details)


def render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Технический разбор реальных примеров Stage 1",
        "",
        "Статус: воспроизводимый технический разбор ограниченной выборки. Это **не "
        "экспертная разметка**, не подтверждение физической поломки и не оценка "
        "медицинской/промышленной безопасности. `candidate` означает только "
        f"срабатывание правила `{result['ruleset']}`; `no_candidate` не доказывает "
        "исправность.",
        "",
        "## Покрытие",
        "",
        f"Разобраны все **{summary['examples']} каналов** текущего каталога: "
        f"**{summary['types']} типов**, **{summary['reporting_groups']} согласованных "
        "групп**. Решения: "
        + ", ".join(f"`{key}` — {value}" for key, value in summary["decisions"].items())
        + ". Выбор примеров после просмотра результатов не выполнялся.",
        f"Всего сохранено **{summary['episodes']} результатов-эпизодов**; их решения: "
        + ", ".join(f"`{key}` — {value}" for key, value in summary["episode_decisions"].items())
        + ".",
        "",
        "Во всех примерах контекст объекта недоступен: в локальном справочнике нет "
        "подтверждённой связи `канал → объект`. Ожидаемая частота регистрации также "
        "не подтверждена, поэтому статус наблюдаемости преимущественно `unknown` и "
        "не переопределяет решение baseline-детектора.",
        "",
        "## Все 38 примеров",
        "",
    ]
    for group, _ in REPORT_GROUPS.items():
        lines.extend(
            [
                f"### {group}",
                "",
                "| ID | Тип / канал | Режим | Решение | Evidence | Observation quality | Контекст и граница | Исходные строки |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for example in result["examples"]:
            if example["report_group"] != group:
                continue
            lines.append(
                "| {example_id} | {sensor_type} / `{channel_id}` | `{processing_mode}` | "
                "`{decision}` | {reason} | {quality} | объект `unknown`; {limit} | {rows} |".format(
                    reason=_reason(example).replace("|", "\\|"),
                    quality=_quality(example).replace("|", "\\|"),
                    limit=example["interpretation_limit"].replace("|", "\\|"),
                    rows=_short_rows(example).replace("|", "\\|"),
                    **example,
                )
            )
        lines.append("")

    unknown_reasons = summary["unknown_detector_reasons"]
    lines.extend(
        [
            "## Почему получен `unknown`",
            "",
            *[f"- `{reason}` — {count} канал(а/ов)." for reason, count in unknown_reasons.items()],
            "",
            "`blocking_quality:channel_time_conflict` означает, что у канала в одну "
            "секунду сохранены разные значения: пайплайн не выбирает одно из них "
            "молча. `insufficient_*_history` означает, что до точки проверки нет "
            "минимального числа пригодных наблюдений. Это причины воздержаться от "
            "решения, а не признаки неисправности.",
            "",
            "## Три сквозных маршрута",
            "",
        ]
    )
    for decision in ("candidate", "no_candidate", "unknown"):
        example = next(item for item in result["examples"] if item["decision"] == decision)
        row = next(
            (
                item
                for item in example["source_rows"]
                if item["relation_to_episode"] == "episode_interval"
            ),
            example["source_rows"][0],
        )
        episode = example["episode"]
        lines.extend(
            [
                f"### {example['example_id']}: `{decision}`",
                "",
                f"1. Исходная ссылка: `{row['source']}:{row['source_row']}`, событие "
                f"`{row['event_id']}`, время `{row['timestamp']}`, исходное значение "
                f"`{row['raw_value']}`, флаги качества `{row['quality_flags']}`, роль "
                f"строки `{row['relation_to_episode']}`.",
                "2. Нормализация сохранила исходный текст и отдельно числовое значение; "
                f"режим обработки — `{example['processing_mode']}`.",
                f"3. Детектор `{episode['anomaly_type']}` сформировал основания: "
                f"`{_reason(example)}`.",
                f"4. Получен Episode `{episode['episode_id']}`: начало "
                f"`{episode['start_at']}`, подтверждение `{episode['confirmed_at']}`, "
                f"решение `{decision}`.",
                f"5. Ограничение: {example['interpretation_limit']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Машиночитаемая детализация",
            "",
            "Полные записи находятся в `output/stage1/real_examples.json`. Для каждого "
            "примера там сохранены до шести ссылок `source + source_row`, исходные "
            "значения, доступный контекст, сводка канала, evidence, качество "
            "наблюдаемости, полный Episode и граница интерпретации. Команды полного "
            "повторного запуска и контрольные суммы описаны в "
            "`docs/stage1-reproducibility.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check", action="store_true", help="Validate inputs without writing")
    args = parser.parse_args()
    result = build_examples(args.catalog.resolve(), args.events.resolve())
    if not args.check:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(result), encoding="utf-8")
    print(
        json.dumps(
            {**result["summary"], "validation": result["validation"]}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
