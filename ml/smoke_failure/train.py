"""Train CatBoost with chronological validation and a locked final test set."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import catboost
from catboost import CatBoostClassifier
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset import CAT_FEATURES, HORIZON_HOURS, MAX_GAP_HOURS, META_COLUMNS, TARGET, build_dataset
from extract import OUT, extract, source_signature

DEFAULT_STUDY = {
    "years": [2022, 2023],
    "training_years": [2022],
    "validation_period": ["2023-01-01", "2023-07-01"],
    "test_period": ["2023-07-01", "2024-01-01"],
    "source_description": "Использованы архивы 2022 и 2023 годов. Пример журнала не используется.",
    "split_description": "Обучение: 2022 год. Валидация: январь–июнь 2023. Тест: июль–декабрь 2023.",
    "plot_title": "Final test: July-December 2023",
}


def save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def choose_threshold(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    candidates = np.flatnonzero((precision > 0.7) & (recall > 0))
    if len(candidates):
        order = np.lexsort((precision[candidates], recall[candidates]))
        index = candidates[order[-1]]
        rule = "maximize validation recall subject to precision > 0.7"
    else:
        index = int(np.argmax(f1))
        rule = "precision > 0.7 unattainable on validation; maximize validation F1"
    return float(thresholds[index]), rule


def classification_metrics(y, score, threshold):
    y = np.asarray(y, dtype=int)
    predicted = np.asarray(score) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = float(precision_score(y, predicted, zero_division=0))
    recall = float(recall_score(y, predicted, zero_division=0))
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "average_precision": float(average_precision_score(y, score)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "precision": precision,
        "recall": recall,
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "requirements_met_on_proxy": bool(precision > 0.7 and recall > 0.5),
    }


def operational_metrics(frame, score, threshold):
    data = frame[["channel_id", "timestamp", TARGET, "next_onset"]].copy()
    data["risk_score"] = score
    positives = data.loc[data[TARGET].eq(1), ["channel_id", "next_onset"]].drop_duplicates()
    warnings = data.loc[data.risk_score.ge(threshold)].sort_values(["channel_id", "timestamp"])
    kept = []
    for _, group in warnings.groupby("channel_id"):
        last = None
        for row in group.itertuples():
            if last is None or row.timestamp - last >= pd.Timedelta(hours=24):
                kept.append(row.Index)
                last = row.timestamp
    alerts = warnings.loc[kept].copy()
    hits = alerts.loc[alerts[TARGET].eq(1)].copy()
    detected = hits.groupby(["channel_id", "next_onset"], as_index=False).timestamp.min()
    lead = (detected.next_onset - detected.timestamp).dt.total_seconds() / 3600
    false_alerts = int(alerts[TARGET].eq(0).sum())
    exposure_days = len(data) / 24
    result = {
        "cooldown_hours": 24,
        "evaluable_onsets": len(positives),
        "detected_onsets": len(detected),
        "episode_recall": float(len(detected) / len(positives)) if len(positives) else None,
        "alerts": len(alerts),
        "false_alerts": false_alerts,
        "alert_precision": float(len(hits) / len(alerts)) if len(alerts) else 0.0,
        "observed_channel_days": exposure_days,
        "false_alerts_per_1000_observed_channel_days": false_alerts / exposure_days * 1000
        if exposure_days
        else None,
        "median_lead_hours": float(lead.median()) if len(lead) else None,
        "min_lead_hours": float(lead.min()) if len(lead) else None,
        "max_lead_hours": float(lead.max()) if len(lead) else None,
    }
    return result, alerts


def render_report(metrics, audit, importance, output, study=None):
    study = study or DEFAULT_STUDY
    test = metrics["test"]
    operational = metrics["test_operational"]
    lines = [
        "# CatBoost: прогноз неисправности дымовых датчиков",
        "",
        f"**Объём проверки: {test['rows']} часовых точек и "
        f"{operational['evaluable_onsets']} доступных эпизодов неисправности. "
        "Метрики относятся к наблюдаемым участкам журнала и производной метке.**",
        "",
        "Модель прогнозирует начало зарегистрированного состояния «Неисправен» в следующие 24 часа. "
        "Метка не подтверждает физическую поломку; дым и пожар не являются целевыми событиями.",
        "",
        "## Источники и разметка",
        "",
        study["source_description"]
        + " Отобраны каналы с типом «Датчик дыма» в предоставленном справочнике. "
        "Историческая неизменность справочника не подтверждена.",
        "",
        "Состояния «Норма», «Дыма нет», «Обнаружен дым» считаются наблюдаемыми состояниями без зарегистрированной неисправности. "
        "Переход из одного из них в «Неисправен» образует начало эпизода, если между наблюдениями не более 24 часов. "
        "Повторные «Неисправен» без промежуточного состояния без неисправности не образуют новый эпизод.",
        "",
        "Прогноз строится раз в час по истории не короче 24 часов, только при последнем известном состоянии без неисправности "
        "и возрасте последнего сообщения не более 24 часов. При текущем «Неисправен» прогноз не выдаётся.",
        "",
        "Для ОБОИХ классов требуется наблюдаемость всего окна (t, t+24ч]. "
        "Метка 1: подтверждённый по журналу переход наступил в (t, t+24ч] внутри непрерывного наблюдаемого участка. "
        "Метка 0: тот же участок покрывает весь горизонт и перехода нет. "
        "Противоречащие состояния в одну секунду, неизвестные состояния, разрывы свыше 24 часов и конец истории цензурируют метку. "
        "При неизвестной периодичности журнала даже это правило является допущением о наблюдаемости; оно не доказывает полноту регистрации.",
        "",
        f"Канонических событий: {audit['canonical_events']:,}; противоречивых временных точек: {audit['ambiguous_timestamps']:,}. "
        f"Пригодных по прошлому часовых точек: {audit['candidate_hours']:,}; без надёжной будущей метки: {audit['censored_hours']:,}. "
        f"Размечено: {audit['labelled_hours']:,} точек у {audit['labelled_channels']:,} каналов.",
        "",
        "## Разделение по времени",
        "",
        study["split_description"] + " Перед границами удалён полный горизонт в 24 часа. "
        "Признаки используют только события с временем <= t; будущие события используются исключительно для разметки. "
        "Идентификатор канала, дата будущего отказа и другие служебные поля исключены из признаков. "
        "Каналы могут встречаться в нескольких периодах: проверяется перенос во времени, а не только на новые датчики.",
        "",
        "| Период | Часовых точек | Положительных | Доступных эпизодов | Каналов с положительными метками |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        s = audit["splits"][split]
        lines.append(
            f"| {split} | {s['rows']:,} | {s['positive_rows']:,} | {s['positive_episodes']:,} | {s['positive_channels']:,} |"
        )
    lines += [
        "",
        "## Качество на окончательном тесте",
        "",
        f"Порог {metrics['threshold']:.6f} выбран только по валидации: {metrics['threshold_rule']}.",
        "",
        "| Метрика | CatBoost | Простое правило по неисправностям за 7 дней |",
        "|---|---:|---:|",
    ]
    for key in ("precision", "recall", "f1", "average_precision", "roc_auc"):
        v, b = test[key], metrics["baseline_test"][key]
        lines.append(
            f"| {key} | {v:.4f} | {b:.4f} |"
            if v is not None and b is not None
            else f"| {key} | {v} | {b} |"
        )
    lines += [
        "",
        f"TP={test['tp']}, FP={test['fp']}, FN={test['fn']}, TN={test['tn']}. "
        f"Доля положительного класса: {test['positive_rate']:.4%}; это ориентир Average Precision случайного ранжирования. "
        "Average Precision вычислена sklearn и является ступенчатой оценкой площади под PR-кривой.",
        "",
        f"Требования Precision > 0,7 и Recall > 0,5 на этой производной цели: "
        f"{'численно выполнены' if test['requirements_met_on_proxy'] else 'НЕ выполнены'}. "
        "Достаточность независимых эпизодов для подтверждения требований не установлена.",
        "",
        "Сравнение с простым правилом приведено выше; статистическое превосходство отдельно не проверялось. "
        "Порог и модель после просмотра теста не изменялись.",
        "",
        "## Предупреждения и упреждение",
        "",
        "Повторное предупреждение одного канала подавляется на 24 часа. Оценка проводится только на часах с известной меткой; "
        "она не описывает нагрузку на весь парк датчиков с пропусками.",
        "",
        f"Обнаружено эпизодов: {operational['detected_onsets']} из {operational['evaluable_onsets']}. "
        f"Предупреждений: {operational['alerts']}, ложных: {operational['false_alerts']}. "
        f"Ложных предупреждений на 1000 наблюдаемых канал-дней: {operational['false_alerts_per_1000_observed_channel_days']:.2f}. "
        f"Медиана упреждения: {operational['median_lead_hours']} ч.",
        "",
        "Горизонт «в ближайшие 24 часа» не гарантирует предупреждение минимум за 24 часа. "
        "Для такого требования нужна другая разметка, например событие в интервале (t+24ч, t+48ч].",
        "",
        "## Признаки",
        "",
        "Текущее состояние и тревога; время с последнего события, смены состояния и сообщения о неисправности; "
        "число событий, тревог, неисправностей, переходов, обнаружений дыма и неизвестных состояний за 1, 24 и 168 часов; "
        "доли тревог и неопределённости, интервалы между сообщениями, календарные признаки. "
        "Выход CatBoost используется как оценка риска; отдельная калибровка вероятностей не проводилась.",
        "",
        "Важность признаков описывает использование моделью и не доказывает причины отказа.",
        "",
        "| Признак | Важность |",
        "|---|---:|",
    ]
    for row in importance.head(12).itertuples():
        lines.append(f"| {row.feature} | {row.importance:.3f} |")
    lines += [
        "",
        "## Ограничения",
        "",
        "Часовые точки коррелируют внутри одного эпизода: число строк не равно числу независимых отказов. "
        "Из-за цензурирования выборка смещена к активно наблюдаемым каналам. "
        "Отсутствуют результаты проверки, ремонты и доказанная разметка ложных срабатываний. "
        "Метрики на зарегистрированных неисправностях нельзя переносить на физические отказы без независимой проверки.",
        "",
        f"Обучение заняло {metrics['training_seconds']:.2f} с; прогноз для {test['rows']:,} тестовых строк "
        f"занял {metrics['test_prediction_seconds']:.3f} с (без построения признаков). "
        "Это исследовательская модель, не подтверждённая для эксплуатационного применения.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def train(data, audit, output=OUT, study=None):
    study = study or DEFAULT_STUDY
    output.mkdir(parents=True, exist_ok=True)
    features = [c for c in data.columns if c not in META_COLUMNS]
    splits = {s: data.loc[data.split.eq(s)].copy() for s in ("train", "validation", "test")}
    for name, frame in splits.items():
        if frame[TARGET].nunique() != 2:
            raise ValueError(f"{name}: cannot train/evaluate binary model without both classes")
        audit["splits"][name]["positive_channels"] = int(
            frame.loc[frame[TARGET].eq(1), "channel_id"].nunique()
        )
    save_json(output / "dataset_audit.json", audit)
    model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="SqrtBalanced",
        random_seed=42,
        thread_count=6,
        allow_writing_files=False,
    )
    started = time.perf_counter()
    model.fit(
        splits["train"][features],
        splits["train"][TARGET],
        cat_features=CAT_FEATURES,
        eval_set=(splits["validation"][features], splits["validation"][TARGET]),
        early_stopping_rounds=80,
        verbose=100,
    )
    training_seconds = time.perf_counter() - started
    model.save_model(str(output / "catboost_smoke_failure.cbm"))
    validation_score = model.predict_proba(splits["validation"][features])[:, 1]
    threshold, rule = choose_threshold(splits["validation"][TARGET], validation_score)
    prediction_start = time.perf_counter()
    test_score = model.predict_proba(splits["test"][features])[:, 1]
    prediction_seconds = time.perf_counter() - prediction_start
    baseline = {
        s: g.faults_168h.to_numpy() / (g.events_168h.to_numpy() + 1) for s, g in splits.items()
    }
    base_threshold, base_rule = choose_threshold(
        splits["validation"][TARGET], baseline["validation"]
    )
    operational, alerts = operational_metrics(splits["test"], test_score, threshold)
    metrics = {
        "target": TARGET,
        "threshold": threshold,
        "threshold_rule": rule,
        "training_seconds": training_seconds,
        "test_prediction_seconds": prediction_seconds,
        "tree_count": model.tree_count_,
        "best_iteration": model.get_best_iteration(),
        "validation": classification_metrics(
            splits["validation"][TARGET], validation_score, threshold
        ),
        "test": classification_metrics(splits["test"][TARGET], test_score, threshold),
        "test_operational": operational,
        "baseline_threshold": base_threshold,
        "baseline_rule": base_rule,
        "baseline_test": classification_metrics(
            splits["test"][TARGET], baseline["test"], base_threshold
        ),
        "monthly_test": {},
    }
    predictions = splits["test"][["channel_id", "timestamp", TARGET, "next_onset"]].copy()
    predictions["risk_score"] = test_score
    predictions["warning"] = test_score >= threshold
    predictions.to_parquet(output / "test_predictions.parquet", index=False)
    predictions.head(1000).to_csv(output / "test_predictions_example.csv", index=False)
    alerts.to_csv(output / "test_alerts.csv", index=False)
    for month, indices in predictions.groupby(
        predictions.timestamp.dt.to_period("M")
    ).groups.items():
        part = predictions.loc[indices]
        metrics["monthly_test"][str(month)] = classification_metrics(
            part[TARGET], part.risk_score, threshold
        )
    importance = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    schema = {
        "target": TARGET,
        "features": features,
        "categorical_features": CAT_FEATURES,
        "threshold": threshold,
        "horizon_hours": HORIZON_HOURS,
        "max_gap_hours": MAX_GAP_HOURS,
        "minimum_history_hours": 24,
        "sensor_type": "Датчик дыма",
        "training_years": study["training_years"],
        "validation_period": study["validation_period"],
        "test_period": study["test_period"],
        "model_sha256": hashlib.sha256(
            (output / "catboost_smoke_failure.cbm").read_bytes()
        ).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "catboost": catboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "parameters": model.get_all_params(),
        "sources": source_signature(study["years"]),
        "evaluation_status": "exploratory_insufficient_independent_test_episodes"
        if operational["evaluable_onsets"] < 30
        else "exploratory_proxy_temporal_test",
        "complete_24h_followup_required_for_both_classes": True,
        "feature_builder_sha256": hashlib.sha256(
            Path(__file__).with_name("dataset.py").read_bytes()
        ).hexdigest(),
    }
    save_json(output / "model_metadata.json", schema)
    save_json(output / "metrics.json", metrics)
    reloaded = CatBoostClassifier()
    reloaded.load_model(str(output / "catboost_smoke_failure.cbm"))
    np.testing.assert_allclose(
        reloaded.predict_proba(splits["test"][features].head(200))[:, 1],
        test_score[:200],
        rtol=0,
        atol=1e-12,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, score in [("CatBoost", test_score), ("Past faults baseline", baseline["test"])]:
        p, r, _ = precision_recall_curve(splits["test"][TARGET], score)
        axes[0].plot(r, p, label=name)
    axes[0].axhline(
        metrics["test"]["positive_rate"], color="gray", linestyle="--", label="Prevalence"
    )
    axes[0].scatter(
        [metrics["test"]["recall"]],
        [metrics["test"]["precision"]],
        color="black",
        label="Locked threshold",
    )
    axes[0].set(
        xlabel="Recall", ylabel="Precision", title=study["plot_title"], xlim=(0, 1), ylim=(0, 1.02)
    )
    axes[0].legend(fontsize=8)
    top = importance.head(12).iloc[::-1]
    axes[1].barh(top.feature, top.importance, color="#16877d")
    axes[1].set(xlabel="CatBoost feature importance", title="Most used features")
    fig.savefig(output / "evaluation.png", dpi=160)
    plt.close(fig)
    render_report(metrics, audit, importance, output, study)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild labels and features from cached smoke events",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "training_dataset.parquet"
    manifest_path = OUT / "dataset_manifest.json"
    signature = {
        "sources": source_signature(),
        "code": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("dataset.py", "extract.py")
        },
    }
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    )
    if args.rebuild or not path.exists() or signature != previous:
        events = extract()
        data, episodes, audit = build_dataset(events)
        data.to_parquet(path, index=False)
        episodes.to_csv(OUT / "fault_episodes.csv", index=False)
        save_json(OUT / "dataset_audit.json", audit)
        save_json(manifest_path, signature)
    else:
        data = pd.read_parquet(path)
        audit = json.loads((OUT / "dataset_audit.json").read_text(encoding="utf-8"))
    train(data, audit)


if __name__ == "__main__":
    main()
