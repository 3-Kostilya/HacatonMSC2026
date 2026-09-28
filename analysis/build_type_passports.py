"""Build reproducible passports for all sensor types in the local dictionary.

The exact yearly counts come from the full, previously generated streaming profiles
in ``analysis/results``.  Value examples are collected from the complete example CSV
and from a bounded prefix of every locally available 7z archive.  The distinction is
kept explicit in the output: sampled values are never presented as exhaustive.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
EVENT_COLUMNS = [
    "ид_события",
    "ид_канала_данных",
    "дата",
    "время",
    "тревожное",
    "значение_датчика",
]

TYPE_META = {
    "Датчик температуры": (
        "Числовые измерения среды",
        "измерение температуры и технические состояния",
        "числовой ряд + отдельная последовательность состояний",
        "Единица в данных не указана; вероятно °C, но это требует подтверждения.",
    ),
    "Газовый датчик": (
        "Числовые измерения среды",
        "измерение газовой среды",
        "числовой ряд по подтверждённым веществу и единицам",
        "Газ и единица не указаны; сравнение каналов до их уточнения недопустимо.",
    ),
    "Датчик дыма": (
        "Пожарные извещатели",
        "дискретное обнаружение дыма и технические состояния",
        "последовательность состояний",
        "Срабатывание означает наблюдение дыма, а не доказанный отказ извещателя.",
    ),
    "Тепловой датчик": (
        "Пожарные извещатели",
        "дискретный тепловой извещатель",
        "последовательность состояний",
        "Не смешивать с числовым датчиком температуры.",
    ),
    "Датчик движения": (
        "Охранные и контактные каналы",
        "обнаружение движения",
        "частота и последовательность событий",
        "Активность зависит от режима доступа и охраны, которого в журнале недостаточно.",
    ),
    "КД АВ": (
        "Охранные и контактные каналы",
        "контактный/охранный канал неуточнённого назначения",
        "частота и последовательность событий",
        "Расшифровка сокращения и физический смысл не подтверждены.",
    ),
    "КД Дверь": (
        "Охранные и контактные каналы",
        "контроль состояния двери",
        "частота и последовательность событий",
        "Смысл состояния контакта и нормальное расписание доступа не заданы.",
    ),
    "КД Люк": (
        "Охранные и контактные каналы",
        "контроль состояния люка",
        "частота и последовательность событий",
        "Смысл состояния контакта и нормальное расписание доступа не заданы.",
    ),
    "9-секционный люк": (
        "Охранные и контактные каналы",
        "контроль многосекционного люка",
        "частота и последовательность событий",
        "События есть только в 2025–2026 годах; устройство и кодирование секций не описаны.",
    ),
    "Стекло": (
        "Охранные и контактные каналы",
        "охранный канал контроля стекла",
        "частота и последовательность событий",
        "Физический принцип и трактовка состояний не описаны.",
    ),
    "Датчик затопления": (
        "Контроль затопления",
        "обнаружение воды/затопления",
        "редкие дискретные события и наблюдаемость",
        "Сигнал воды не доказывает отказ датчика; малая выборка ограничивает профиль.",
    ),
    "Состояние насоса": (
        "Состояние оборудования",
        "наблюдаемый статус насосного оборудования",
        "циклы и частота переключений",
        "Канал состояния не тождественен насосу; причина изменения не дана.",
    ),
    "Состояние вентилятора": (
        "Состояние оборудования",
        "наблюдаемый статус вентиляционного оборудования",
        "циклы и частота переключений",
        "Канал состояния не тождественен вентилятору; причина изменения не дана.",
    ),
    "ИБП": (
        "Питание и состояние аппаратуры",
        "состояние и отдельные числовые сообщения ИБП",
        "состояния + изолированный анализ числовых сообщений",
        "Физический смысл и единицы чисел не указаны.",
    ),
    "Состояние фазы": (
        "Питание и состояние аппаратуры",
        "наличие питания/состояние фазы",
        "последовательность состояний и контекст общих событий",
        "Топология питания отсутствует; совместность не доказывает общую причину.",
    ),
    "Состояние УИР-Р": (
        "Питание и состояние аппаратуры",
        "техническое состояние устройства неуточнённой роли",
        "последовательность состояний после уточнения семантики",
        "Расшифровка УИР-Р и смысл состояний не подтверждены.",
    ),
    "Переключатель": (
        "Управление, ручные сигналы и режим охраны",
        "управляющее переключение",
        "контекстная последовательность действий",
        "Команда оператора или автоматики не должна автоматически считаться отказом.",
    ),
    "Ручной извещатель": (
        "Управление, ручные сигналы и режим охраны",
        "ручной сигнал извещателя",
        "редкие контекстные события",
        "Действие человека не является самостоятельной меткой отказа устройства.",
    ),
    "Состояние охраны": (
        "Управление, ручные сигналы и режим охраны",
        "режим охраны и сводные сообщения",
        "контекст для охранных событий",
        "Встречаются значения, похожие на даты/время; их семантика не подтверждена.",
    ),
}


def parse_number(value: str) -> float | None:
    """Return a finite float for unambiguous numeric strings."""
    try:
        number = float(value.strip())
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def display_number(value: object) -> str:
    if value is None:
        return "—"
    number = float(value)
    return f"{number:.6g}"


def pct(part: int, total: int) -> str:
    if not total:
        return "—"
    percent = 100 * part / total
    if part and percent < 0.01:
        return "<0.01%"
    return f"{percent:.2f}%"


def md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def find_7z(explicit: str | None) -> str | None:
    candidates = [
        explicit,
        os.environ.get("SEVEN_ZIP"),
        shutil.which("7z"),
        r"C:\Program Files\7-Zip\7z.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def read_dictionary(
    path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    by_channel: dict[str, dict[str, str]] = {}
    by_type: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"ид_канала_данных", "тип_инж_системы", "тип_датчика"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected dictionary schema: {reader.fieldnames}")
        for row in reader:
            channel = row["ид_канала_данных"]
            if channel in by_channel:
                raise ValueError(f"Duplicate channel ID in dictionary: {channel}")
            by_channel[channel] = row
            by_type[row["тип_датчика"]].append(row)
    return by_channel, dict(by_type)


def load_profiles(directory: Path) -> dict[int, dict]:
    profiles = {}
    for path in sorted(directory.glob("ext-journal-*.json")):
        try:
            year = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        profiles[year] = json.loads(path.read_text(encoding="utf-8"))
    if not profiles:
        raise FileNotFoundError(f"No full profiles found in {directory}")
    return profiles


def empty_sample() -> dict:
    return {
        "rows": 0,
        "numeric": collections.Counter(),
        "text": collections.Counter(),
    }


def sample_rows(
    rows: Iterable[dict[str, str]],
    channel_map: dict[str, dict[str, str]],
    year_hint: int | None,
    samples: dict[tuple[str, int], dict],
    limit: int | None,
) -> int:
    seen = 0
    for row in rows:
        if limit is not None and seen >= limit:
            break
        seen += 1
        dictionary_row = channel_map.get(row.get("ид_канала_данных", ""))
        if dictionary_row is None:
            continue
        raw_year = (row.get("дата") or "")[:4]
        year = int(raw_year) if raw_year.isdigit() else year_hint
        if year is None:
            continue
        value = row.get("значение_датчика", "")
        kind = "numeric" if parse_number(value) is not None else "text"
        bucket = samples.setdefault((dictionary_row["тип_датчика"], year), empty_sample())
        bucket["rows"] += 1
        bucket[kind][value] += 1
    return seen


def sample_csv(
    path: Path,
    channel_map: dict[str, dict[str, str]],
    samples: dict[tuple[str, int], dict],
) -> int:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return sample_rows(csv.DictReader(stream), channel_map, None, samples, None)


def sample_archive(
    path: Path,
    seven_zip: str,
    channel_map: dict[str, dict[str, str]],
    samples: dict[tuple[str, int], dict],
    limit: int,
) -> int:
    process = subprocess.Popen(
        [seven_zip, "x", "-so", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if process.stdout is None:
        raise RuntimeError(f"Cannot read {path}")
    text = io.TextIOWrapper(process.stdout, encoding="utf-8-sig", newline="")
    try:
        reader = csv.DictReader(text)
        if reader.fieldnames != EVENT_COLUMNS:
            raise ValueError(f"Unexpected archive schema in {path}: {reader.fieldnames}")
        return sample_rows(reader, channel_map, int(path.stem[-4:]), samples, limit)
    finally:
        # A bounded prefix intentionally closes the decompressor before archive EOF.
        try:
            text.detach()
        except ValueError:
            pass
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)


def exact_type_year(
    sensor_type: str,
    year: int,
    profile: dict,
    channel_map: dict[str, dict[str, str]],
) -> dict[str, object]:
    events = int(profile.get("types", {}).get(sensor_type, 0))
    numeric_info = profile.get("numeric_by_type", {}).get(sensor_type, {})
    numeric = int(numeric_info.get("count", 0))
    active = 0
    for channel, count in profile.get("channels", {}).items():
        row = channel_map.get(channel)
        if count and row and row["тип_датчика"] == sensor_type:
            active += 1
    return {
        "year": year,
        "events": events,
        "active_channels": active,
        "numeric": numeric,
        "text": events - numeric,
        "numeric_min": numeric_info.get("min"),
        "numeric_max": numeric_info.get("max"),
    }


def top_values(bucket: dict | None, kind: str, limit: int = 5) -> str:
    if not bucket:
        return "—"
    values = bucket[kind].most_common(limit)
    return "; ".join(md(value) for value, _ in values) or "—"


def build_report(
    by_channel: dict[str, dict[str, str]],
    by_type: dict[str, list[dict[str, str]]],
    profiles: dict[int, dict],
    samples: dict[tuple[str, int], dict],
    archive_years: set[int],
    sampled_sources: list[str],
) -> tuple[str, list[dict[str, object]]]:
    unknown_meta = set(by_type) - set(TYPE_META)
    missing_types = set(TYPE_META) - set(by_type)
    if unknown_meta or missing_types:
        raise ValueError(f"Type metadata mismatch: unknown={unknown_meta}, missing={missing_types}")

    years = sorted(profiles)
    exact_rows = []
    for sensor_type in sorted(by_type):
        for year in years:
            row = exact_type_year(sensor_type, year, profiles[year], by_channel)
            row["sensor_type"] = sensor_type
            row["dictionary_channels"] = len(by_type[sensor_type])
            row["active_share"] = (
                row["active_channels"] / row["dictionary_channels"]
                if row["dictionary_channels"]
                else None
            )
            exact_rows.append(row)

    lines = [
        "# Паспорта типов каналов",
        "",
        "Сформировано `analysis/build_type_passports.py` по локальному справочнику и "
        "полногодовым потоковым профилям журналов. Точные объёмы, активные каналы и "
        "доли форматов ниже относятся ко всему профилированному году. Примеры значений "
        "получены из ограниченной потоковой выборки и **не являются полным словарём состояний**.",
        "",
        "## Источники и воспроизводимость",
        "",
        f"- Справочник: {len(by_channel):,} каналов, {len(by_type)} типов; колонка объекта отсутствует.",
        f"- Полные профили: {', '.join(map(str, years))}. Они дают точные агрегаты по всем строкам.",
        "- Локальные исходные архивы: "
        + (", ".join(map(str, sorted(archive_years))) or "нет")
        + ".",
        "- Архив 2021 года локально отсутствует: его агрегаты доступны только в ранее сохранённом "
        "полном профиле и сейчас не воспроизводятся из сырого файла.",
        "- Потоковые источники примеров: " + "; ".join(sampled_sources) + ".",
        "- `Неисправен` по QA означает наблюдаемую потерю связи, а не подтверждённую физическую поломку.",
        "- Режим регистрации (периодический опрос или запись изменений) неизвестен, поэтому отсутствие "
        "строк и длительность состояния нельзя трактовать безусловно.",
        "- Воспроизведение: `python -X utf8 analysis/build_type_passports.py`. Для быстрого "
        "пересчёта без выборки архивов: `--archive-sample-rows 0`.",
        "",
        "## Статус семантики значений",
        "",
        "| Класс значения | Что известно | Что остаётся unknown |",
        "|---|---|---|",
        "| `Неисправен` | По QA — потеря связи | Физическая причина, виновное устройство и факт поломки |",
        "| `Норма`, `Неопределен` | Реальные названия состояний журнала | Критерий присвоения и длительность состояния |",
        "| Предметные пары (`Включен`/`Выключен`, `Есть питание`/`Обесточен`, "
        "`Обнаружено движение`/`Движения нет`) | Буквальный наблюдаемый статус | Режим регистрации, "
        "нормальный рабочий режим и связь с отказом |",
        "| Числа | Формат, диапазон и частота по типу/году | Единицы, физическая величина отдельных "
        "каналов и перечень технических кодов |",
        "| Строки, похожие на дату/время | Встречаются у состояния охраны | Назначение; считаются "
        "неинтерпретированным значением, а не временем события |",
        "",
        "## Сводный охват",
        "",
        "| Тип | Группа | Каналов в справочнике | Годы с событиями | Всего событий | Формат |",
        "|---|---|---:|---|---:|---|",
    ]
    for sensor_type in sorted(by_type):
        rows = [r for r in exact_rows if r["sensor_type"] == sensor_type]
        observed = [str(r["year"]) for r in rows if r["events"]]
        events = sum(int(r["events"]) for r in rows)
        numeric = sum(int(r["numeric"]) for r in rows)
        if not events:
            form = "нет наблюдений"
        elif not numeric:
            form = "текстовый"
        elif numeric == events:
            form = "числовой"
        else:
            form = f"смешанный ({pct(numeric, events)} чисел)"
        lines.append(
            f"| {md(sensor_type)} | {md(TYPE_META[sensor_type][0])} | "
            f"{len(by_type[sensor_type]):,} | {', '.join(observed) or '—'} | {events:,} | {form} |"
        )

    lines.extend(
        [
            "",
            "## Матрица применимости признаков",
            "",
            "`Да` означает базовый признак; `условно` — только после проверки семантики или как "
            "контекст; `нет` — неприменимо как основной признак.",
            "",
            "| Тип | Уровень/разброс числа | Состояния/переходы | Частота событий | Совместность |",
            "|---|---|---|---|---|",
        ]
    )
    numeric_primary = {"Датчик температуры", "Газовый датчик"}
    numeric_conditional = {"ИБП"}
    state_conditional = {"Датчик температуры", "Газовый датчик", "Состояние УИР-Р"}
    frequency_conditional = {"Датчик температуры", "Газовый датчик", "ИБП"}
    for sensor_type in sorted(by_type):
        numeric = (
            "да"
            if sensor_type in numeric_primary
            else "условно"
            if sensor_type in numeric_conditional
            else "нет (числа — контроль качества)"
        )
        states = "условно" if sensor_type in state_conditional else "да"
        frequency = "условно" if sensor_type in frequency_conditional else "да"
        lines.append(
            f"| {md(sensor_type)} | {numeric} | {states} | {frequency} | "
            "условно, без вывода о причине |"
        )

    lines.extend(["", "## Паспорта", ""])
    for sensor_type in sorted(by_type):
        group, role, detector, limitation = TYPE_META[sensor_type]
        dictionary_rows = by_type[sensor_type]
        systems = sorted({r["тип_инж_системы"] for r in dictionary_rows})
        rows = [r for r in exact_rows if r["sensor_type"] == sensor_type]
        sampled_text = collections.Counter()
        sampled_numeric = collections.Counter()
        for year in years:
            bucket = samples.get((sensor_type, year))
            if bucket:
                sampled_text.update(bucket["text"])
                sampled_numeric.update(bucket["numeric"])
        lines.extend(
            [
                f"### {sensor_type}",
                "",
                f"- **Группа / роль:** {group}; {role}.",
                f"- **Справочник:** {len(dictionary_rows):,} каналов; инженерная система: "
                f"{', '.join(systems)}.",
                f"- **Применимый способ:** {detector}.",
                "- **Наблюдаемые текстовые состояния (выборка):** "
                + (", ".join(md(v) for v, _ in sampled_text.most_common(12)) or "не встретились")
                + ".",
                "- **Наблюдаемые числовые значения (выборка):** "
                + (", ".join(md(v) for v, _ in sampled_numeric.most_common(8)) or "не встретились")
                + ".",
                f"- **Ограничения / unknown:** {limitation} Единицы, технические коды, "
                "историческая замена устройства и физическая привязка к объекту в справочнике не заданы.",
                "",
                "| Год | Активных каналов | Покрытие справочника | Событий | Числа | Текст | "
                "Диапазон чисел | Примеры из потоковой выборки |",
                "|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in rows:
            bucket = samples.get((sensor_type, int(row["year"])))
            examples = []
            text = top_values(bucket, "text", 3)
            numeric_values = top_values(bucket, "numeric", 3)
            if text != "—":
                examples.append("текст: " + text)
            if numeric_values != "—":
                examples.append("числа: " + numeric_values)
            numeric_range = "—"
            if row["numeric"]:
                numeric_range = (
                    f"{display_number(row['numeric_min'])}…{display_number(row['numeric_max'])}"
                )
            lines.append(
                f"| {row['year']} | {row['active_channels']:,} | "
                f"{pct(int(row['active_channels']), len(dictionary_rows))} | {row['events']:,} | "
                f"{row['numeric']:,} ({pct(int(row['numeric']), int(row['events']))}) | "
                f"{row['text']:,} | {numeric_range} | {'; '.join(examples) or '—'} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Общие ограничения",
            "",
            "1. Справочник отражает 11 485 записей каналов, но не доказывает число физических "
            "устройств и не содержит ID объекта.",
            "2. Текущий справочник применён к историческим ID. Замены и изменение назначения канала "
            "не описаны; доля активных каналов — покрытие текущего справочника, а не инвентаризация года.",
            "3. Текстовые примеры взяты из ограниченного префикса архивов и полного файла-примера; "
            "редкие состояния могли не попасть в паспорт.",
            "4. Числовой формат не доказывает физическую величину. Особенно это относится к ИБП, "
            "редким числам дискретных типов и техническим кодам температуры.",
            "5. Тревожный флаг, `Неисправен` и алгоритмическая аномалия — разные признаки; ни один "
            "из них сам по себе не подтверждает физическую поломку.",
            "6. До уточнения режима регистрации оцениваются наблюдаемые события, а не непрерывная "
            "длительность состояний.",
            "",
        ]
    )
    return "\n".join(lines), exact_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sensor_type",
        "year",
        "dictionary_channels",
        "active_channels",
        "active_share",
        "events",
        "numeric",
        "text",
        "numeric_min",
        "numeric_max",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-sample-rows",
        type=int,
        default=250_000,
        help="Maximum prefix rows sampled from each local archive; 0 disables archive sampling.",
    )
    parser.add_argument("--seven-zip", help="Path to 7z executable (or set SEVEN_ZIP).")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "type-passports.md")
    parser.add_argument(
        "--csv-output", type=Path, default=ROOT / "output" / "stage1" / "type-year-coverage.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dictionary_path = ROOT / "data" / "справочник_каналов_датчиков.csv"
    sample_path = ROOT / "data" / "журнал_событий_пример.csv"
    by_channel, by_type = read_dictionary(dictionary_path)
    profiles = load_profiles(ROOT / "analysis" / "results")
    archives = sorted((ROOT / "data").glob("ext-journal-*.7z"))
    archive_years = {int(path.stem[-4:]) for path in archives}
    samples: dict[tuple[str, int], dict] = {}
    sampled_sources = []
    if sample_path.is_file():
        count = sample_csv(sample_path, by_channel, samples)
        sampled_sources.append(f"полный файл-пример ({count:,} строк)")
    if args.archive_sample_rows:
        seven_zip = find_7z(args.seven_zip)
        if archives and seven_zip is None:
            raise FileNotFoundError("7z not found; pass --seven-zip or use --archive-sample-rows 0")
        for archive in archives:
            count = sample_archive(
                archive, seven_zip or "", by_channel, samples, args.archive_sample_rows
            )
            sampled_sources.append(f"{archive.name}: первые {count:,} строк")
    report, rows = build_report(
        by_channel, by_type, profiles, samples, archive_years, sampled_sources
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    write_csv(args.csv_output, rows)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
