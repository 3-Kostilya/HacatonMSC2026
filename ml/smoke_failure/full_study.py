"""The same smoke-fault experiment over all annual archives, 2019-2026."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import (
    AMBIGUOUS,
    HORIZON_HOURS,
    MAX_GAP_HOURS,
    TARGET,
    canonicalize,
    features_at,
    label_at,
    timeline,
)
from extract import ROOT, extract, source_signature
from train import classification_metrics, operational_metrics, save_json, train

OUTPUT = ROOT / "output" / "smoke_failure_2019_2026"
STUDY = {
    "years": list(range(2019, 2027)),
    "training_years": list(range(2019, 2024)),
    "validation_period": ["2024-01-01", "2025-01-01"],
    "test_period": ["2025-01-01", "2026-07-01"],
    "source_description": "Использованы все восемь годовых архивов 2019–2026. Архив 2026 заканчивается 30 июня. Отдельный пример за август 2026 не включён.",
    "split_description": "Обучение: 2019–2023. Валидация и выбор порога: 2024. Окончательный тест: 2025 и первое полугодие 2026.",
    "plot_title": "Final test: 2025 - June 2026",
}


def split_full(frame):
    frame = frame.copy()
    t = frame.timestamp
    end = t + pd.Timedelta(hours=HORIZON_HOURS)
    frame["split"] = "excluded"
    for split, start, stop in [
        ("train", "2019-01-01", "2024-01-01"),
        ("validation", "2024-01-01", "2025-01-01"),
        ("test", "2025-01-01", "2026-07-01"),
    ]:
        frame.loc[t.ge(start) & end.lt(pd.Timestamp(stop)), "split"] = split
    return frame


def build_full(events):
    canonical = canonicalize(events)
    parts, episodes, year_stats = [], [], {}
    audit = {
        "canonical_events": len(canonical),
        "ambiguous_timestamps": int(canonical.state.eq(AMBIGUOUS).sum()),
        "max_observation_gap_hours": MAX_GAP_HOURS,
        "horizon_hours": HORIZON_HOURS,
        "prediction_step_hours": 1,
        "candidate_hours": 0,
        "censored_hours": 0,
        "channels_with_eligible_hours": 0,
        "study": STUDY,
    }
    for year in STUDY["years"]:
        group = canonical.loc[canonical.timestamp.dt.year.eq(year)]
        year_stats[str(year)] = {
            "canonical_events": len(group),
            "channels": int(group.channel_id.nunique()),
            "start": str(group.timestamp.min()),
            "end": str(group.timestamp.max()),
            "candidate_hours": 0,
            "censored_hours": 0,
        }
    groups = canonical.groupby("channel_id", sort=True)
    for i, (channel, group) in enumerate(groups):
        group = group.reset_index(drop=True)
        line = timeline(group)
        episodes.extend(
            {"channel_id": channel, "onset": group.timestamp.iloc[idx]}
            for idx in np.flatnonzero(line["onset"])
        )
        start = (group.timestamp.iloc[0] + pd.Timedelta(hours=24)).ceil("h")
        end = min(
            group.timestamp.iloc[-1] + pd.Timedelta(hours=24),
            pd.Timestamp("2026-07-01") - pd.Timedelta(hours=1),
        )
        if start > end:
            continue
        features = features_at(group, pd.date_range(start, end, freq="h"))
        if features.empty:
            continue
        labels = label_at(group, features.timestamp)
        audit["candidate_hours"] += len(features)
        audit["censored_hours"] += int(labels[TARGET].eq(-1).sum())
        audit["channels_with_eligible_hours"] += 1
        frame = pd.concat([features.reset_index(drop=True), labels], axis=1)
        for year, yearly in frame.groupby(frame.timestamp.dt.year):
            year_stats[str(year)]["candidate_hours"] += len(yearly)
            year_stats[str(year)]["censored_hours"] += int(yearly[TARGET].eq(-1).sum())
        parts.append(frame.loc[frame[TARGET].ge(0)].copy())
        if i % 250 == 0:
            print(f"Full history: {i + 1}/{len(groups)} channels", flush=True)
    data = (
        split_full(pd.concat(parts, ignore_index=True))
        .sort_values(["timestamp", "channel_id"])
        .reset_index(drop=True)
    )
    episodes = pd.DataFrame(episodes, columns=["channel_id", "onset"])
    audit.update(
        {
            "confirmed_onsets": len(episodes),
            "labelled_hours": len(data),
            "labelled_channels": int(data.channel_id.nunique()),
            "years": year_stats,
        }
    )
    for year, g in data.groupby(data.timestamp.dt.year):
        positives = g.loc[g[TARGET].eq(1)]
        year_stats[str(year)].update(
            {
                "labelled_hours": len(g),
                "positive_rows": int(g[TARGET].sum()),
                "positive_channels": int(positives.channel_id.nunique()),
                "positive_episodes": len(positives[["channel_id", "next_onset"]].drop_duplicates()),
            }
        )
    audit["splits"] = {
        split: {
            "rows": len(g),
            "positive_rows": int(g[TARGET].sum()),
            "channels": int(g.channel_id.nunique()),
            "positive_episodes": len(
                g.loc[g[TARGET].eq(1), ["channel_id", "next_onset"]].drop_duplicates()
            ),
            "start": str(g.timestamp.min()),
            "end": str(g.timestamp.max()),
        }
        for split, g in data.groupby("split")
    }
    return data, episodes, audit


def extend_report(data, metrics, audit):
    predictions = pd.read_parquet(OUTPUT / "test_predictions.parquet")
    metrics["yearly_test"] = {}
    for year, group in predictions.groupby(predictions.timestamp.dt.year):
        m = classification_metrics(group[TARGET], group.risk_score, metrics["threshold"])
        m["operational"], _ = operational_metrics(group, group.risk_score, metrics["threshold"])
        metrics["yearly_test"][str(year)] = m
    save_json(OUTPUT / "metrics.json", metrics)
    rows = []
    lines = [
        "",
        "## Покрытие всех годов",
        "",
        "Соседние годовые архивы соединены до построения истории. Положительные и отрицательные метки требуют одинакового полного окна наблюдений. "
        "Пропущенные дни и повторный заголовок в архиве 2025 не трактуются как наблюдения исправного датчика.",
        "",
        "| Год | Событий дымовых каналов после объединения | Каналов | Кандидатов | Размечено | Положительных | Эпизодов в метках |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for year, s in audit["years"].items():
        row = {"year": int(year), **s}
        rows.append(row)
        lines.append(
            f"| {year} | {s['canonical_events']:,} | {s['channels']:,} | {s['candidate_hours']:,} | "
            f"{s.get('labelled_hours', 0):,} | {s.get('positive_rows', 0):,} | {s.get('positive_episodes', 0):,} |"
        )
    yearly = pd.DataFrame(rows)
    yearly.to_csv(OUTPUT / "yearly_coverage.csv", index=False)
    lines += [
        "",
        "Число эпизодов в годовой таблице относится к времени прогноза; один эпизод на границе года может учитываться в двух строках. "
        "Год 2026 содержит только первое полугодие, поэтому сравнение абсолютных объёмов с полными годами ограничено.",
        "",
        "## Проверка отдельно по годам",
        "",
        "| Год | Примеров | Precision | Recall | Average Precision | Эпизодов | Найдено с паузой 24 ч |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for year, m in metrics["yearly_test"].items():
        op = m["operational"]
        lines.append(
            f"| {year} | {m['rows']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['average_precision']} | "
            f"{op['evaluable_onsets']} | {op['detected_onsets']} |"
        )
    positive = data.loc[data[TARGET].eq(1)]
    channel_counts = (
        positive.groupby(["split", "channel_id"]).size().rename("positive_hours").reset_index()
    )
    channel_counts.sort_values(["split", "positive_hours"], ascending=[True, False]).to_csv(
        OUTPUT / "positive_channel_concentration.csv", index=False
    )
    lines += [
        "",
        "## Концентрация положительных примеров",
        "",
        "| Период | Доля самого частого канала | Доля пяти самых частых |",
        "|---|---:|---:|",
    ]
    for split, g in channel_counts.groupby("split"):
        counts = g.positive_hours.sort_values(ascending=False)
        lines.append(
            f"| {split} | {counts.iloc[0] / counts.sum():.2%} | {counts.head(5).sum() / counts.sum():.2%} |"
        )
    lines += [
        "",
        "## Сопоставление с исследованием 2022–2023",
        "",
        "В предыдущем эксперименте тест состоял из 73 часовых точек и одного эпизода. "
        "Сейчас используются другие периоды обучения и проверки, поэтому изменение метрик нельзя приписывать только росту объёма обучения. "
        "Сохранены прежние правила разметки, признаки, настройки CatBoost и способ выбора порога. "
        "Случайного перемешивания временных выборок нет; данные 2025–2026 не участвуют в выборе числа деревьев и порога.",
        "",
    ]
    report = OUTPUT / "report.md"
    report.write_text(report.read_text(encoding="utf-8") + "\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(yearly.year.astype(str), yearly.canonical_events, color="#16877d")
    axes[0].set(title="Smoke-channel events by year", ylabel="Events")
    axes[1].bar(yearly.year.astype(str), yearly.labelled_hours.fillna(0), color="#c55165")
    axes[1].set(title="Hours with observable 24h labels", ylabel="Labelled hours")
    fig.savefig(OUTPUT / "yearly_coverage.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    signature = {
        "sources": source_signature(STUDY["years"]),
        "study": STUDY,
        "code": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("dataset.py", "extract.py", "full_study.py")
        },
    }
    manifest_path = OUTPUT / "dataset_manifest.json"
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    )
    path = OUTPUT / "training_dataset.parquet"
    if args.rebuild or signature != previous or not path.exists():
        event_path = OUTPUT / "events.parquet"
        extraction_path = OUTPUT / "extraction_audit.json"
        extraction_audit = (
            json.loads(extraction_path.read_text(encoding="utf-8"))
            if extraction_path.exists()
            else {}
        )
        if event_path.exists() and extraction_audit.get("sources") == signature["sources"]:
            events = pd.read_parquet(event_path)
        else:
            events = extract(STUDY["years"], OUTPUT)
        data, episodes, audit = build_full(events)
        data.to_parquet(path, index=False)
        episodes.to_csv(OUTPUT / "fault_episodes.csv", index=False)
        save_json(OUTPUT / "dataset_audit.json", audit)
        save_json(manifest_path, signature)
    else:
        data = pd.read_parquet(path)
        audit = json.loads((OUTPUT / "dataset_audit.json").read_text(encoding="utf-8"))
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    metrics = train(data, audit, OUTPUT, STUDY)
    extend_report(data, metrics, audit)


if __name__ == "__main__":
    main()
