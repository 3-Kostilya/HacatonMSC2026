"""Render warning precision/full recall curves for the fixed Q3 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def plot(experiment: Path, output: Path) -> None:
    report = json.loads((experiment / "report.json").read_text(encoding="utf-8"))
    policies = {"base": ("Q2", "#2166ac"), "cold_start": ("Без ожидания", "#e08214"),
                "after_normal": ("После Норма", "#1b9e77"), "combined": ("Оба изменения", "#88419d")}
    markers = {"base51": "o", "full121": "s", "linear121": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    left, right = axes
    left.fill_between([0.5, 1], 0.7, 1, color="#d9efdc", alpha=0.8)
    left.text(0.68, 0.83, "Область целей", color="#236b36")
    for policy, stage in report["comparison_2023"].items():
        for family, model in stage["models"].items():
            curve = model["curve"]
            left.plot([row["matched_episodes"] / stage["all_assigned_episodes"] for row in curve],
                      [row["episode_precision"] for row in curve], color=policies[policy][1],
                      marker=markers[family], markersize=4, linewidth=1, alpha=0.75)
    handles = [Line2D([], [], color=color, label=name) for name, color in policies.values()]
    handles += [Line2D([], [], color="#555555", marker=marker, linestyle="", label=family)
                for family, marker in markers.items()]
    left.legend(handles=handles, loc="upper left", fontsize=8)
    left.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall от всех эпизодов",
             ylabel="Precision предупреждений", title="2023: 12 вариантов, сетка порогов")
    chosen = report["decision_2023"]["candidate_for_2024"]
    checked = report["confirmation_2024"]["models"][chosen["model"]]["candidate"]
    for offset, field, color, label in ((-0.18, "episode_precision", "#2166ac", "Precision"),
                                      (0.18, "full_episode_recall", "#e08214", "Полный Recall")):
        bars = right.bar([offset, 1 + offset], [chosen[field], checked[field]], width=0.34,
                         color=color, label=label)
        right.bar_label(bars, labels=[f"{chosen[field]:.1%}", f"{checked[field]:.1%}"], padding=3)
    right.axhline(0.7, color="#2166ac", linestyle="--", linewidth=1)
    right.axhline(0.5, color="#e08214", linestyle="--", linewidth=1)
    right.set(xticks=[0, 1], xticklabels=["Выбор: 2023", "Проверка: 2024"], ylim=(0, 1),
              title=f"{chosen['policy']} / {chosen['model']}\nОдин числовой порог: {chosen['threshold']:.6g}")
    right.legend(loc="upper right", fontsize=9)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Q3/B: ретроспективное исследование, предупреждение ↔ эпизод")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot(args.experiment, args.output)
