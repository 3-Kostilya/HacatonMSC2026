"""Summarize existing journal profiles for scenario selection without unpacking archives."""

import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "analysis/results"
OUT = ROOT / "output/data_quality"
TYPES = [
    "Датчик температуры",
    "Состояние насоса",
    "Датчик движения",
    "КД Дверь",
    "Датчик дыма",
    "Датчик затопления",
]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(numerator, denominator):
    return f"{100 * numerator / denominator:.2f}%" if denominator else "—"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dictionary = ROOT / "data/справочник_каналов_датчиков.csv"
    dictionary_audit = load(ROOT / "output/dictionary_check/audit.json")
    if (
        hashlib.sha256(dictionary.read_bytes()).hexdigest()
        != dictionary_audit["channels"]["sha256"]
    ):
        raise ValueError("Channel dictionary changed since the dictionary audit")
    years = [load(SAVED / f"ext-journal-{year}.json") for year in range(2019, 2027)]
    for year, saved in zip(range(2019, 2027), years, strict=True):
        archive = ROOT / f"data/ext-journal-{year}.7z"
        if archive.stat().st_size != saved["bytes"] or saved["archive_exit_code"] != 0:
            raise ValueError(f"Journal profile is stale: {archive.name}")
    with dictionary.open(encoding="utf-8-sig", newline="") as stream:
        channel_types = {
            row["ид_канала_данных"]: row["тип_датчика"] for row in csv.DictReader(stream)
        }
    recent_profiles = {year: years[year - 2019] for year in (2024, 2025, 2026)}
    continuity = {}
    for kind in TYPES:
        ids = {channel for channel, sensor_type in channel_types.items() if sensor_type == kind}
        active = {
            year: {channel for channel in ids if profile["channels"].get(channel, 0) > 0}
            for year, profile in recent_profiles.items()
        }
        frequent = {
            year: {
                channel
                for channel in ids
                if profile["channels"].get(channel, 0) >= (50 if year == 2026 else 100)
            }
            for year, profile in recent_profiles.items()
        }
        continuity[kind] = {
            "active_by_year": {year: len(channels) for year, channels in active.items()},
            "active_all_three_years": len(set.intersection(*active.values())),
            "at_least_100_100_50_events": len(set.intersection(*frequent.values())),
        }

    yearly = []
    for year, item in zip(range(2019, 2027), years, strict=True):
        alarms = item["alarm"].get("t", 0)
        yearly.append(
            {
                "year": year,
                "rows": item["rows"],
                "alarm_rows": alarms,
                "alarm_rate_percent": 100 * alarms / item["rows"],
                "failure_rows": item["values"].get("Неисправен", 0),
                "unknown_channel_rows": item["unknown_rows"],
                "duplicate_event_ids": item["duplicate_event_ids"],
                "duplicate_full_rows_hash64": item["duplicate_full_rows_hash64"],
                "invalid_date_rows": item["invalid_date_rows"],
                "invalid_time_rows": item["invalid_time_rows"],
                "negative_numeric_rows": item["negative_numeric_rows"],
                "type_rows": {kind: item["types"].get(kind, 0) for kind in TYPES},
                "failure_by_type": {kind: item["failure_types"].get(kind, 0) for kind in TYPES},
                "numeric_by_type": {kind: item["numeric_by_type"].get(kind) for kind in TYPES},
            }
        )

    total_rows = sum(item["rows"] for item in years)
    totals = {
        "rows": total_rows,
        "alarms": sum(item["alarm"].get("t", 0) for item in years),
        "failure_rows": sum(item["values"].get("Неисправен", 0) for item in years),
        "negative_numeric_rows": sum(item["negative_numeric_rows"] for item in years),
        "duplicate_event_ids_within_years": sum(item["duplicate_event_ids"] for item in years),
        "duplicate_full_rows_hash64_within_years": sum(
            item["duplicate_full_rows_hash64"] for item in years
        ),
        "invalid_date_rows": sum(item["invalid_date_rows"] for item in years),
        "invalid_time_rows": sum(item["invalid_time_rows"] for item in years),
        "type_rows": dict(
            collections.Counter(
                {kind: sum(item["types"].get(kind, 0) for item in years) for kind in TYPES}
            )
        ),
    }
    model_audit = load(ROOT / "output/smoke_failure_2019_2026/dataset_audit.json")
    extraction_audit = load(ROOT / "output/smoke_failure_2019_2026/extraction_audit.json")
    sample_comparison = load(SAVED / "sample_comparison.json")
    machine = {
        "method": "Saved annual profiles; no archive decompression or model training",
        "years": yearly,
        "totals": totals,
        "continuity": continuity,
        "smoke_model_audit": {
            key: model_audit[key]
            for key in [
                "candidate_hours",
                "censored_hours",
                "labelled_hours",
                "labelled_channels",
                "ambiguous_timestamps",
            ]
        },
        "smoke_raw_rows": extraction_audit["raw_smoke_rows"],
        "sample_id_collisions_with_2024": sum(
            row["matched_rows"] for row in sample_comparison["years"]
        ),
        "sample_exact_matches_after_normalization": sum(
            row["fully_matching_rows_normalized_alarm"] for row in sample_comparison["years"]
        ),
    }
    (OUT / "audit.json").write_text(
        json.dumps(machine, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Качество журналов и выбор первого сценария",
        "",
        "Проверка использует готовый анализ всех восьми годовых архивов и предыдущую проверку справочника. Архивы не распаковывались повторно, модель не переобучалась. Размеры архивов сверены с сохранёнными профилями, текущий справочник — по SHA-256. Метрики относятся к записям журналов, а не к числу подтверждённых физических аварий.",
        "",
        "## Главные ограничения",
        "",
        f"- Всего {total_rows:,} записей. Из них {totals['alarms']:,} помечены как тревожные ({pct(totals['alarms'], total_rows)}). Флаг тревоги и значение «Неисправен» не являются подтверждением аварии.",
        f"- В текущем справочнике нет ID объекта. {dictionary_audit['total']['unknown_rows']:,} событий с числовым ID канала ({dictionary_audit['total']['unknown_channels']:,} каналов) не имеют записи в справочнике; основная часть относится к ранним годам.",
        f"- {totals['duplicate_event_ids_within_years']:,} повторов ID события внутри годовых выгрузок. Это количество повторных вхождений ID, а не число уникальных конфликтующих ID. {totals['duplicate_full_rows_hash64_within_years']:,} совпадений хеша полной строки; коллизии хеша отдельно не исключались. Склеивать события только по ID нельзя.",
        f"- В примере журнала {machine['sample_id_collisions_with_2024']} ID пересекаются с архивом 2024 года, но совпадений всех полей нет. Пример нельзя прибавлять к историческим архивам как независимые записи.",
        f"- Отрицательных числовых значений {totals['negative_numeric_rows']:,}; особенно много в 2020–2021 годах. Для каждого типа и производителя нужно отделять допустимые значения от технических кодов.",
        f"- Ошибка даты и времени встречается в одной строке 2025 года ({totals['invalid_date_rows']} и {totals['invalid_time_rows']} соответственно). Неполные дни есть в 2024 и 2026 годах; архив 2026 заканчивается 30 июня.",
        "",
        "## Изменение по годам",
        "",
        "| Год | Записей, млн | Тревог | Доля тревог | «Неисправен» | Повторы ID события |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in yearly:
        lines.append(
            f"| {row['year']} | {row['rows'] / 1e6:.2f} | {row['alarm_rows']:,} | "
            f"{row['alarm_rate_percent']:.2f}% | {row['failure_rows']:,} | "
            f"{row['duplicate_event_ids']:,} |"
        )
    lines += [
        "",
        "В 2021 году значение «Неисправен» встречается резко чаще, чем в соседние годы. Спикеры связали массовые тревоги этого периода с переходом между системами. При оценке любого сценария нужен отдельный результат с 2021 годом и без него. Сравнивать 2026 год как полный год нельзя.",
        "",
        "## Кандидаты для MVP",
        "",
        "Наличие записей за год не доказывает непрерывность измерений. В следующей таблице «100/100/50» означает не менее 100 событий в каждом из 2024 и 2025 годов и не менее 50 в первом полугодии 2026 года; это только быстрый фильтр объёма истории.",
        "",
        "| Тип датчика | Активны в каждом из 2024–2026 | Проходят фильтр 100/100/50 | Доля числовых значений 2024 / 2025 / 2026 |",
        "|---|---:|---:|---:|",
    ]
    for kind in TYPES:
        fractions = [
            pct(
                recent_profiles[year]["numeric_by_type"].get(kind, {}).get("count", 0),
                recent_profiles[year]["types"].get(kind, 0),
            )
            for year in recent_profiles
        ]
        lines.append(
            f"| {kind} | {continuity[kind]['active_all_three_years']:,} | "
            f"{continuity[kind]['at_least_100_100_50_events']:,} | {' / '.join(fractions)} |"
        )
    lines += [
        "",
        "У температурных датчиков доля числовых показаний снизилась с 80.60% в 2024 году до 50.39% в первом полугодии 2026 года. Часть записей содержит текстовые состояния. Общая доля не показывает, сколько конкретных каналов сохранили полезный числовой ряд; это надо проверить отдельно перед выбором модели.",
        "",
        "| Сценарий | Каналов в текущем справочнике | Записей за 2019–июнь 2026 | Что позволяет журнал | Главный пробел |",
        "|---|---:|---:|---|---|",
    ]
    scenario_rows = [
        (
            "Температурная аномалия",
            "Датчик температуры",
            "Числовой ряд и тревожные состояния; можно показывать тенденцию по каналу.",
            "Есть значения от −3276 до 999 в 2025 году: требуется разбор технических кодов и порогов. Нет связи с объектом.",
        ),
        (
            "Частое переключение насоса",
            "Состояние насоса",
            "Состояния насоса и временные метки позволяют искать частые переходы.",
            "Нет подтверждённых неисправностей, ремонта и топологии; причину переключений нельзя установить по одному журналу.",
        ),
        (
            "Последовательность охранных событий",
            "Датчик движения",
            "Есть много событий движения; можно искать повторяющиеся последовательности по каналу.",
            "Связь с дверью и соседними датчиками через объект отсутствует; надёжно подтвердить проникновение нельзя.",
        ),
        (
            "Прогноз состояния дымового датчика",
            "Датчик дыма",
            "Есть действующий исследовательский процесс и подготовленная модель.",
            "Строгая разметка оставила лишь 5 222 часовых примера; результат не подтверждает физическую поломку.",
        ),
    ]
    sensor_counts = dictionary_audit["channels"]["sensor_types"]
    for name, kind, supported, gap in scenario_rows:
        lines.append(
            f"| {name} | {sensor_counts[kind]:,} | {totals['type_rows'][kind]:,} | "
            f"{supported} | {gap} |"
        )
    lines += [
        "",
        "**Рекомендация:** первым кандидатом для нового сценария взять температурную аномалию на уровне одного канала. У неё есть числовые показания, поэтому можно демонстрировать изменение во времени и объяснение предупреждения без недоступной связи с объектом. Решение о сценарии принять после проверки, сколько каналов сохраняют непрерывный числовой ряд в 2025–2026 годах: общая доля числовых значений падает. Для насоса сделать следующий исследовательский проход; он особенно интересен заказчику, но требует аккуратного определения повторного включения и исключения нормального режима работы.",
        "",
        f"Существующая модель дымовых датчиков остаётся отдельным экспериментом: из {machine['smoke_raw_rows']:,} исходных записей получено {model_audit['labelled_hours']:,} размеченных часовых примера ({model_audit['labelled_channels']} каналов). Её метрики нельзя переносить на все датчики или подтверждённые аварии.",
        "",
        "## Следующий конкретный шаг",
        "",
        "Построить профиль температурных каналов за 2024–2026 годы: доля числовых и текстовых значений, технические коды, частота и длина разрывов, число каналов с устойчивой историей. После этого выбрать правило предупреждения и временной тест. Эта проверка потребует чтения относящихся к температуре записей в архивах: сохранённые профили дают диапазоны и частоты по типу, но не полную последовательность каждого канала.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
