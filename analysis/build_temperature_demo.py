"""Build an explainable low-temperature warning demonstration for one channel."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/temperature_anomaly"
CHANNEL_ID = "286947"
WATCH_MIN = 3.0
WATCH_MAX = 5.0
MAX_PREVIOUS_GAP_HOURS = 24
COOLDOWN_HOURS = 48


def build():
    source = OUT / "stable_temperature_events.parquet"
    raw = pd.read_parquet(source)
    cohort = raw.copy()
    cohort["timestamp"] = pd.to_datetime(cohort["дата"] + " " + cohort["время"])
    cohort["number"] = pd.to_numeric(cohort["значение_датчика"], errors="coerce")
    comparison = []
    for channel_id, group in cohort.loc[cohort["number"].notna()].groupby("ид_канала_данных"):
        series = group.drop_duplicates(["timestamp", "number"]).sort_values("timestamp")
        values = series.set_index("timestamp")["number"]
        rolling_range = values.rolling("30min").max() - values.rolling("30min").min()
        days_with_fast_swing = int(
            rolling_range.ge(10).groupby(rolling_range.index.normalize()).any().sum()
        )
        comparison.append(
            {
                "channel_id": channel_id,
                "numeric_rows": len(series),
                "days_with_10c_swing_within_30min": days_with_fast_swing,
            }
        )
    pd.DataFrame(comparison).sort_values(["days_with_10c_swing_within_30min", "channel_id"]).to_csv(
        OUT / "candidate_comparison.csv", index=False, encoding="utf-8-sig"
    )
    data = raw.loc[raw["ид_канала_данных"].eq(CHANNEL_ID)].copy()
    data["timestamp"] = pd.to_datetime(data["дата"] + " " + data["время"])
    data["number"] = pd.to_numeric(data["значение_датчика"], errors="coerce")
    data = data.sort_values(["timestamp", "ид_события"]).reset_index(drop=True)
    numeric = data.loc[data["number"].notna()].copy()
    numeric = numeric.drop_duplicates(["timestamp", "number"])
    numeric = numeric.sort_values("timestamp")
    numeric["gap_hours"] = numeric["timestamp"].diff().dt.total_seconds() / 3600

    warnings = []
    last_warning = None
    for row in numeric.itertuples():
        eligible = (
            WATCH_MIN < row.number <= WATCH_MAX
            and pd.notna(row.gap_hours)
            and row.gap_hours <= MAX_PREVIOUS_GAP_HOURS
            and (
                last_warning is None
                or row.timestamp - last_warning >= pd.Timedelta(hours=COOLDOWN_HOURS)
            )
        )
        if eligible:
            warnings.append(
                {
                    "channel_id": CHANNEL_ID,
                    "timestamp": row.timestamp,
                    "temperature_c": row.number,
                    "previous_measurement_gap_hours": round(row.gap_hours, 3),
                    "reason": "Температура приблизилась к порогу 3°C",
                }
            )
            last_warning = row.timestamp
    alerts = pd.DataFrame(warnings)
    alerts.to_csv(OUT / "warnings_286947.csv", index=False, encoding="utf-8-sig")

    critical = data.loc[data["значение_датчика"].eq("Температура ниже 3ºC"), "timestamp"]
    critical = critical.sort_values().tolist()
    episodes = []
    for timestamp in critical:
        if not episodes or timestamp - episodes[-1][-1] > pd.Timedelta(hours=24):
            episodes.append([timestamp])
        else:
            episodes[-1].append(timestamp)
    episode_checks = []
    for group in episodes:
        onset = group[0]
        prior = alerts.loc[
            alerts["timestamp"].lt(onset) & alerts["timestamp"].ge(onset - pd.Timedelta(hours=48))
        ]
        if prior.empty:
            lead = None
            warning_at = None
        else:
            warning_at = prior.iloc[-1]["timestamp"]
            lead = round((onset - warning_at).total_seconds() / 3600, 2)
        episode_checks.append(
            {
                "critical_at": onset.isoformat(),
                "text_alarm_rows": len(group),
                "preceding_warning_at": warning_at.isoformat() if warning_at is not None else None,
                "lead_hours": lead,
            }
        )

    numeric_low = numeric.loc[numeric["number"].lt(3), "timestamp"].tolist()
    numeric_episodes = []
    for timestamp in numeric_low:
        if not numeric_episodes or timestamp - numeric_episodes[-1][-1] > pd.Timedelta(hours=24):
            numeric_episodes.append([timestamp])
        else:
            numeric_episodes[-1].append(timestamp)
    numeric_episode_checks = []
    for group in numeric_episodes:
        onset = group[0]
        prior = alerts.loc[
            alerts["timestamp"].lt(onset) & alerts["timestamp"].ge(onset - pd.Timedelta(hours=48))
        ]
        warning_at = None if prior.empty else prior.iloc[-1]["timestamp"]
        numeric_episode_checks.append(
            {
                "first_below_3_at": onset.isoformat(),
                "preceding_warning_at": warning_at.isoformat() if warning_at is not None else None,
                "lead_hours": round((onset - warning_at).total_seconds() / 3600, 2)
                if warning_at is not None
                else None,
            }
        )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Температурный канал 286947: показания и предупреждения", fontsize=16)
    periods = [
        ("2025-01-01", "2026-07-01", "Вся доступная история"),
        ("2026-01-20", "2026-02-11", "Период зарегистрированных тревог"),
    ]
    for ax, (start, end, title) in zip(axes, periods, strict=True):
        left, right = pd.Timestamp(start), pd.Timestamp(end)
        points = numeric.loc[numeric["timestamp"].ge(left) & numeric["timestamp"].lt(right)]
        flagged = alerts.loc[alerts["timestamp"].ge(left) & alerts["timestamp"].lt(right)]
        alarms = [stamp for stamp in critical if left <= stamp < right]
        ax.scatter(
            points["timestamp"],
            points["number"],
            s=8,
            alpha=0.65,
            color="#266A9E",
            label="Числовые показания",
        )
        ax.axhspan(3, 5, color="#F3CF7A", alpha=0.45, label="Зона предупреждения 3–5°C")
        ax.axhline(3, color="#B73832", linewidth=1.1, label="Порог журнала 3°C")
        if not flagged.empty:
            ax.scatter(
                flagged["timestamp"],
                flagged["temperature_c"],
                marker="^",
                s=70,
                color="#CB6A00",
                edgecolors="black",
                linewidths=0.3,
                zorder=4,
                label="Наше предупреждение",
            )
        if alarms:
            alarm_numeric = []
            for timestamp in alarms:
                same = points.loc[points["timestamp"].eq(timestamp), "number"]
                alarm_numeric.append(float(same.iloc[0]) if not same.empty else 2.5)
            ax.scatter(
                alarms,
                alarm_numeric,
                marker="x",
                s=75,
                color="#B73832",
                linewidths=2,
                zorder=5,
                label="Тревога в журнале",
            )
        ax.set_xlim(left, right)
        ax.set_ylim(-5, 35)
        ax.set_title(title)
        ax.set_ylabel("Значение, °C")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%y"))
    axes[-1].set_xlabel("Дата")
    fig.savefig(OUT / "temperature_demo_286947.png", dpi=170)
    plt.close(fig)

    result = {
        "channel_id": CHANNEL_ID,
        "numeric_rows": len(numeric),
        "text_alarm_rows": len(critical),
        "text_alarm_episodes_24h": len(episodes),
        "warnings": len(alerts),
        "warnings_2025": int(alerts["timestamp"].dt.year.eq(2025).sum()),
        "warnings_2026_h1": int(alerts["timestamp"].dt.year.eq(2026).sum()),
        "critical_episodes_preceded_within_48h": sum(
            item["lead_hours"] is not None for item in episode_checks
        ),
        "episode_checks": episode_checks,
        "numeric_below_3_episodes_24h": len(numeric_episodes),
        "numeric_below_3_episodes_preceded_within_48h": sum(
            item["lead_hours"] is not None for item in numeric_episode_checks
        ),
        "numeric_episode_checks": numeric_episode_checks,
        "rule": {
            "watch_zone_c": "3 < value <= 5",
            "max_gap_since_previous_numeric_hours": MAX_PREVIOUS_GAP_HOURS,
            "warning_cooldown_hours": COOLDOWN_HOURS,
            "no_measurement_status_after_hours": 24,
        },
        "latest_measurement_at": numeric.iloc[-1]["timestamp"].isoformat(),
        "days_with_fast_swing_2943": next(
            item["days_with_10c_swing_within_30min"]
            for item in comparison
            if item["channel_id"] == "2943"
        ),
        "days_with_fast_swing_286947": next(
            item["days_with_10c_swing_within_30min"]
            for item in comparison
            if item["channel_id"] == CHANNEL_ID
        ),
    }
    (OUT / "demo_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# Первое предупреждение по температурному каналу",
        "",
        f"Использован канал 286947 («ТЕМП ПК388+5») из архивов 2025 года и января–июня 2026-го. Он выбран после сравнения 31 температурного канала: у первоначального кандидата 2943 в {result['days_with_fast_swing_2943']} днях встречались изменения минимум на 10°C в пределах 30 минут, у канала 286947 — в {result['days_with_fast_swing_286947']} днях. Поэтому простое отклонение от средней величины на канале 2943 давало много неубедительных тревог. Сравнение сохранено в `candidate_comparison.csv`.",
        "",
        "## Правило",
        "",
        "Предупреждение появляется при числовом показании выше 3°C и не выше 5°C, если предыдущее числовое показание было не более 24 часов назад. Повторное предупреждение по тому же каналу допускается через 48 часов. Значение ниже 3°C уже относится к критическому состоянию, которое сам журнал обозначает отдельной тревогой. Если после последнего измерения прошло более 24 часов, интерфейс должен показывать «нет свежего измерения» вместо оценки риска.",
        "",
        f"На историческом ряду правило выдало **{len(alerts)} предупреждений**: {result['warnings_2025']} за 2025 год и {result['warnings_2026_h1']} за первое полугодие 2026-го. Числовых эпизодов с показаниями ниже 3°C было {len(numeric_episodes)}; перед {result['numeric_below_3_episodes_preceded_within_48h']} из них было предупреждение за предыдущие 48 часов. В журнале также есть {len(critical)} текстовых тревог, сгруппированных в {len(episodes)} эпизода при паузе 24 часа. Текстовые тревоги не всегда ставятся на первом числовом значении ниже 3°C.",
        "",
        "| Первый числовой показатель ниже 3°C | Предыдущее предупреждение | Упреждение, ч |",
        "|---|---|---:|",
    ]
    for item in numeric_episode_checks:
        report.append(
            f"| {item['first_below_3_at'].replace('T', ' ')} | "
            f"{item['preceding_warning_at'].replace('T', ' ') if item['preceding_warning_at'] else '—'} | "
            f"{item['lead_hours'] if item['lead_hours'] is not None else '—'} |"
        )
    report += [
        "",
        "Это проверка работы правила на одном канале, а не измерение точности прогноза аварий. Порог предупреждения 5°C и целевое условие ниже 3°C измеряются одним датчиком; это не независимая разметка инцидента. Текстовые тревоги формирует та же система мониторинга, и они не подтверждены журналом диспетчера. Порог 5°C — исследовательское значение для раннего внимания, а не установленный заказчиком норматив.",
        "",
        "Файлы: `warnings_286947.csv` — время и причина предупреждений; `temperature_demo_286947.png` — график показаний и тревог; `demo_audit.json` — параметры и проверочные подсчёты.",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"episode_checks", "numeric_episode_checks"}
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    build()
