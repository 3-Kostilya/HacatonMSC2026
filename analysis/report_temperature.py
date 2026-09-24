"""Turn the saved temperature scan into a concise, reproducible decision report."""

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/temperature_profile"


def main():
    audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    yearly = pd.read_csv(OUT / "channel_year.csv", encoding="utf-8-sig", dtype={"channel_id": str})
    with (ROOT / "data/справочник_каналов_датчиков.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        names = {
            row["ид_канала_данных"]: row["название_датчика"].strip()
            for row in csv.DictReader(stream)
        }

    stable = set(audit["stable_channel_ids"])
    selected = yearly.loc[yearly["channel_id"].isin(stable) & yearly["year"].isin([2025, 2026])]
    by_channel = selected.groupby("channel_id").agg(
        review_range_rows=("review_range_rows", "sum"),
        min_numeric_share=("numeric_share", "min"),
    )
    clean = by_channel.index[
        by_channel["review_range_rows"].eq(0) & by_channel["min_numeric_share"].ge(0.8)
    ]
    top = yearly.loc[yearly["channel_id"].isin(clean) & yearly["year"].isin([2025, 2026])].pivot(
        index="channel_id",
        columns="year",
        values=["numeric_rows", "numeric_days", "p90_gap_hours", "numeric_share"],
    )
    top["sort"] = top["p90_gap_hours"][2025] + top["p90_gap_hours"][2026]
    top = top.sort_values("sort").head(5)

    lines = [
        "# Температурные каналы: проверка пригодности",
        "",
        "Проверены все события температурных каналов в архивах 2024, 2025 и января–июня 2026 года. Выборка каналов взята из текущего справочника. Количество отобранных записей и числовых значений по каждому году совпало с сохранённым общим анализом. Файл `channel_year.csv` содержит подробности по каждому каналу и году.",
        "",
        "## Результат",
        "",
        "| Период | Записей | Числовых показаний | Каналов с числами | Медиана дней с числами на канал | Чисел для ручной проверки* |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in audit["year_summary"]:
        lines.append(
            f"| {row['year']}{' (январь–июнь)' if row['year'] == 2026 else ''} | "
            f"{row['rows']:,} | {row['numeric_rows']:,} "
            f"({100 * row['numeric_rows'] / row['rows']:.1f}%) | "
            f"{row['channels_with_numeric']:,} | {row['numeric_days_median_per_channel']:.1f} | "
            f"{row['review_range_rows']:,} |"
        )
    lines += [
        "",
        "*Для ручной проверки отмечены числа ниже −50 или выше 100. Это диагностический диапазон, а не утверждение о физически допустимых температурах. Часто встречается код −127; допустимость других крайних значений зависит от оборудования.",
        "",
        f"У {audit['channels_numeric_100_in_2025_and_50_in_2026']} каналов не менее 100 числовых записей за 2025 год и 50 за первое полугодие 2026-го. У {audit['channels_numeric_days_30_in_2025_and_15_in_2026']} каналов числа встречаются не менее чем в 30 и 15 разных днях соответственно. По всем трём фильтрам, включая 90-й процентиль промежутков между измерениями не более 24 часов в обоих периодах, проходит **{audit['channels_meeting_all_three_filters']} канал**. У {len(clean)} из них числовые записи составляют не менее 80% в каждом периоде и нет чисел за пределами диагностического диапазона.",
        "",
        "Порог 24 часа выбран для отбора каналов с достаточно частыми показаниями, а не задан заказчиком как требование к периодичности датчика. Даже прошедшие фильтр каналы могут иметь отдельные большие разрывы. Временные интервалы нельзя заполнять как реальные измерения.",
        "",
        f"Проверка нашла {audit['duplicate_semantic_rows']:,} полностью повторяющихся событий и {audit['conflicting_channel_timestamps']:,} случаев, когда у одного канала и времени записано более одного значения. Часть таких пар может сочетать числовой показатель с текстовым состоянием; перед построением временного ряда нужно задать правило их обработки. Некорректных временных меток и кодов тревоги среди отобранных записей нет.",
        "",
        "## Каналы для первого прототипа",
        "",
        "Это примеры для изучения, а не доказательство аварий или измерительной точности. Сортировка — по сумме 90-х процентилей промежутков за 2025 и 2026 годы.",
        "",
        "| ID | Название | Числовых показаний 2025 / 2026 | Дней с показаниями 2025 / 2026 | 90% промежутков 2025 / 2026, ч |",
        "|---|---|---:|---:|---:|",
    ]
    for channel, row in top.iterrows():
        lines.append(
            f"| {channel} | {names[channel]} | "
            f"{int(row['numeric_rows'][2025]):,} / {int(row['numeric_rows'][2026]):,} | "
            f"{int(row['numeric_days'][2025]):,} / {int(row['numeric_days'][2026]):,} | "
            f"{row['p90_gap_hours'][2025]:.1f} / {row['p90_gap_hours'][2026]:.1f} |"
        )
    lines += [
        "",
        "## Решение",
        "",
        "Температурный сценарий пригоден для **демонстрации аномалий на отдельных каналах**: есть 30 каналов с частыми числовыми показаниями по принятым фильтрам. Канал 2943 был первым кандидатом по частоте, но детальный просмотр выявил повторяющиеся быстрые проходы значений. Для первого правила выбран канал 286947; результаты — в `output/temperature_anomaly/report.md`. Обучение модели и метрики прогноза подтверждённой аварии пока не обоснованы.",
        "",
        "Для сценария по всему коллектору по-прежнему нужен обновлённый справочник с ID объекта. Этот анализ не устанавливает, что числовые показания всех температурных каналов имеют одинаковые единицы измерения или физический смысл.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
