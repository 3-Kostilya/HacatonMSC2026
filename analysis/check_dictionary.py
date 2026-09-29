"""Audit current dictionaries using saved channel counts, without unpacking journals."""

import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analysis/results"
OUT = ROOT / "output/dictionary_check"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def audit_table(path):
    columns, rows = read_csv(path)
    keys = [r[columns[0]] for r in rows]
    return (
        columns,
        rows,
        {
            "rows": len(rows),
            "columns": columns,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "duplicate_keys": len(keys) - len(set(keys)),
            "duplicate_rows": len(rows) - len({tuple(r.values()) for r in rows}),
            "blank_fields": {c: sum(not r[c].strip() for r in rows) for c in columns},
            "whitespace_fields": {c: sum(r[c] != r[c].strip() for r in rows) for c in columns},
            "invalid_keys": [k for k in keys if not k.isdecimal()],
            "unique": {c: len({r[c] for r in rows}) for c in columns},
        },
    )


def member_metadata(listing):
    return listing.split("----------", 1)[1].strip().replace("\r\n", "\n")


def coverage(counts, known):
    valid = {k: n for k, n in counts.items() if k.isdecimal()}
    matched = sum(n for k, n in valid.items() if k in known)
    return {
        "rows": sum(counts.values()),
        "valid_channel_rows": sum(valid.values()),
        "invalid_channel_rows": sum(n for k, n in counts.items() if not k.isdecimal()),
        "distinct_channels": len(valid),
        "matched_channels": len(set(valid) & known),
        "unknown_channels": len(set(valid) - known),
        "matched_rows": matched,
        "unknown_rows": sum(valid.values()) - matched,
        "matched_rows_percent": 100 * matched / sum(valid.values()),
        "dictionary_channels_without_events": len(known - set(valid)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cc, channels, ca = audit_table(ROOT / "data/справочник_каналов_датчиков.csv")
    _, objects, oa = audit_table(ROOT / "data/справочник_объектов_диспетчер.csv")
    known = {r["ид_канала_данных"] for r in channels}
    by_id = {r["ид_объект"]: r for r in objects}
    ca["object_id_columns"] = [c for c in cc if "объект" in c.lower()]
    ca["sensor_types"] = dict(collections.Counter(r["тип_датчика"] for r in channels))
    oa["missing_parent_rows"] = [r for r in objects if r["родитель"] not in by_id]
    oa["levels"] = dict(collections.Counter(r["иерархия_уровень"] for r in objects))
    oa["level_errors"] = [
        r
        for r in objects
        if r["родитель"] in by_id
        and int(r["иерархия_уровень"]) != int(by_id[r["родитель"]]["иерархия_уровень"]) + 1
    ]
    oa["cycle_start_ids"] = []
    for key in by_id:
        seen = set()
        current = key
        while current in by_id:
            if current in seen:
                oa["cycle_start_ids"].append(key)
                break
            seen.add(current)
            current = by_id[current]["родитель"]

    seven = shutil.which("7z") or shutil.which("7zz") or "C:/Program Files/7-Zip/7z.exe"
    counts = collections.Counter()
    yearly = []
    profiles = []
    for year in range(2019, 2027):
        saved = json.loads((RESULTS / f"ext-journal-{year}.json").read_text(encoding="utf-8"))
        archive = ROOT / "data" / saved["file"]
        listing = subprocess.check_output([seven, "l", "-slt", str(archive)], encoding="utf-8")
        same = archive.stat().st_size == saved["bytes"] and member_metadata(
            listing
        ) == member_metadata(saved["archive_listing"])
        if not same or saved["archive_exit_code"] != 0:
            raise ValueError(f"Cached archive profile cannot be reused: {archive.name}")
        assert sum(saved["channels"].values()) == saved["rows"]
        counts.update(saved["channels"])
        yearly.append({"year": year, **coverage(saved["channels"], known)})
        profiles.append({"file": archive.name, "size_and_member_metadata_match": same})

    _, sample = read_csv(ROOT / "data/журнал_событий_пример.csv")
    sample_counts = collections.Counter(r["ид_канала_данных"] for r in sample)
    sample_saved = json.loads((RESULTS / "журнал_событий_пример.json").read_text(encoding="utf-8"))
    old = json.loads((RESULTS / "dictionaries.json").read_text(encoding="utf-8"))
    previous_match = {}
    for name, audit in [
        ("справочник_каналов_датчиков.csv", ca),
        ("справочник_объектов_диспетчер.csv", oa),
    ]:
        previous_match[name] = all(
            audit[k] == old[name][k] for k in ["rows", "columns", "unique", "duplicate_keys"]
        )
    result = {
        "channels": ca,
        "objects": oa,
        "yearly": yearly,
        "total": coverage(counts, known),
        "sample": coverage(sample_counts, known),
        "cache_checks": profiles,
        "sample_counts_match_previous": dict(sample_counts) == sample_saved["channels"],
        "dictionary_structure_matches_previous": previous_match,
        "cache_limit": "Archive size and stored member metadata including CRC match. Archive contents were not decompressed or rehashed.",
    }
    (OUT / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    unknown = sorted(
        ((k, n) for k, n in counts.items() if k.isdecimal() and k not in known), key=lambda x: -x[1]
    )
    (OUT / "unknown_channels.json").write_text(
        json.dumps(unknown, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = result["total"]
    lines = [
        "# Проверка справочников",
        "",
        "В текущем справочнике каналов отсутствует ID объекта. Достоверная связь канал → объект недоступна. Нужен обновлённый файл от организаторов; технический тег не использован как замена ключа.",
        "",
        "## Методика",
        "",
        "Повторно прочитаны только два справочника и пример журнала. Для восьми годовых архивов использованы сохранённые частоты каналов из analysis/results/ext-journal-*.json. Размеры архивов и метаданные их CSV (включая CRC) совпали с предыдущим анализом. Повторная распаковка и проверка содержимого архивов не проводились. Контрольные суммы текущих справочников сохранены в audit.json; старых контрольных сумм нет.",
        "Пример журнала проверен отдельно и не прибавлен к годовым архивам. Покрытие считается по записям с числовым ID канала; повторные события не удаляются. Нечисловые маркеры выделены отдельно.",
        "",
        "## Справочники",
        "",
        f"- Каналы: {ca['rows']} строк, {len(ca['sensor_types'])} типов датчиков; дубликатов ключей: {ca['duplicate_keys']}; пустых полей: {sum(ca['blank_fields'].values())}.",
        f"- Объекты: {oa['rows']} строк; дубликатов ключей: {oa['duplicate_keys']}; пустых полей: {sum(oa['blank_fields'].values())}.",
        f"- Уровни объектов: {oa['levels']}. Циклов: {len(oa['cycle_start_ids'])}; нарушений перехода уровней по присутствующим родителям: {len(oa['level_errors'])}.",
        f"- Ссылки на отсутствующих родителей: {len(oa['missing_parent_rows'])}. "
        + "; ".join(
            f"{r['ид_объект']} → {r['родитель']} ({r['диспетчерское_название_объекта']})"
            for r in oa["missing_parent_rows"]
        )
        + ". Это может быть внешний корень сокращённой выгрузки, а не ошибка; требуется уточнение.",
        "- Названия датчиков и объектов не уникальны; использовать их как ключи нельзя.",
        f"- У {ca['whitespace_fields']['название_датчика']} названий датчиков есть внешние пробелы. Это косметическая проблема; ID каналов таких пробелов не содержат.",
        "",
        "## Покрытие событий справочником каналов",
        "",
        "| Год | Каналов в журнале | Нет в справочнике | Событий без канала в справочнике | Покрытие событий |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in yearly:
        lines.append(
            f"| {row['year']} | {row['distinct_channels']:,} | {row['unknown_channels']:,} | {row['unknown_rows']:,} | {row['matched_rows_percent']:.5f}% |"
        )
    lines += [
        "",
        f"Всего: {total['rows']:,} записей, {total['distinct_channels']:,} числовых ID каналов. Не найдены {total['unknown_channels']:,} каналов ({total['unknown_rows']:,} событий). Покрытие событий: {total['matched_rows_percent']:.2f}%. Нечисловых ID: {total['invalid_channel_rows']}. Каналов справочника без событий во всех архивах: {total['dictionary_channels_without_events']}.",
        f"Пример журнала: {result['sample']['rows']:,} записей, покрытие {result['sample']['matched_rows_percent']:.2f}%; частоты каналов совпали с предыдущим анализом: {result['sample_counts_match_previous']}.",
        "",
        "## Следующие действия",
        "",
        "1. Получить обновлённый справочник с ID объекта и описанием уровня объекта, на который ссылается ключ.",
        "2. Уточнить полноту исторического справочника для отсутствующих каналов и смысл внешнего родителя 3831.",
        "3. После обновления проверить внешний ключ, покрытие событий по объектам и отсутствие размножения строк при объединении.",
        "4. До обновления продолжать проверку качества на уровне каналов. Сохранённый эксперимент дымовых датчиков использует ID канала и тип датчика; отсутствие ID объекта не отменяет его, но не позволяет достоверно агрегировать по коллекторам.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "total": total,
                "yearly": yearly,
                "channels": ca,
                "objects": oa,
                "previous_match": previous_match,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
