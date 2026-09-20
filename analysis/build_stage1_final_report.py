"""Assemble and verify the final Stage-1 completion and readiness report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.readiness import ReadinessEvidence, assess_readiness  # noqa: E402
from stage1.detectors import RULESET_VERSION  # noqa: E402
from stage1.verification import verify_causal_invariants  # noqa: E402


DEFAULT_OUTPUT = ROOT / "output" / "stage1" / "completion_report.json"
DEFAULT_DOCUMENT = ROOT / "docs" / "stage1-final-report.md"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(benchmark: dict[str, Any], variant: str = "baseline") -> dict[str, Any]:
    return benchmark["variants"][variant]["report"]


def _passed_scenario_ids(checks: dict[str, Any]) -> set[str]:
    """Return only explicitly passed scenario IDs from a complete check record."""

    details = checks.get("details")
    if not isinstance(details, list):
        return set()
    return {
        item.get("scenario_name", item["scenario_id"])
        for item in details
        if isinstance(item, dict)
        and isinstance(item.get("scenario_id"), str)
        and item.get("passed") is True
    }


def _causal_checks_passed(verification: dict[str, Any]) -> bool:
    """Accept causal evidence only when the executable check result is internally complete."""

    checks = verification.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or not all(isinstance(value, bool) for value in checks.values())
    ):
        return False
    passed = sum(checks.values())
    return (
        verification.get("all_passed") is True
        and verification.get("passed") == passed
        and verification.get("total") == len(checks)
        and passed == len(checks)
    )


def _render_blocker(blocker: str) -> str:
    labels = {
        "missing:all_19_types_covered": "не подтверждён охват всех 19 типов",
        "missing:all_7_groups_covered": "не подтверждён охват всех 7 групп",
        "missing:synthetic_protocol_covered": "не все сценарии прошли в обоих наборах",
        "missing:real_examples_reviewed": "недостаточно разобранных реальных примеров",
        "missing:causal_checks_passed": "не пройдены исполняемые причинные проверки",
        "missing:reproducible_run_available": "нет согласованных воспроизводимых артефактов",
        "missing:temporal_protocol_frozen": "не подтверждён зафиксированный временной протокол",
        "missing:candidates_traceable": "не все реальные кандидаты прослеживаемы",
        "missing:full_history_candidate_catalog": "не построен полный исторический каталог кандидатов",
        "missing:candidate_diversity_sufficient": "недостаточно разнообразия кандидатов",
        "missing:external_validation_available": "нет независимой внешней проверки",
    }
    return labels.get(blocker, blocker)


def build_report() -> dict[str, Any]:
    catalog = _read(ROOT / "output" / "stage1" / "baseline_catalog.json")
    examples = _read(ROOT / "output" / "stage1" / "real_examples.json")
    tuning = _read(ROOT / "output" / "stage1" / "tuning_benchmark.json")
    holdout = _read(ROOT / "output" / "stage1" / "synthetic_benchmark.json")
    tuning_metrics = _metrics(tuning)
    holdout_metrics = _metrics(holdout)
    tuning_checks = tuning.get("scenario_checks", {})
    holdout_checks = holdout.get("scenario_checks", {})
    scenario_count = len(_passed_scenario_ids(tuning_checks) & _passed_scenario_ids(holdout_checks))
    reproducibility_files = (
        ROOT / "docs" / "stage1-reproducibility.md",
        ROOT / "output" / "stage1" / "normalized_sample.manifest.json",
        ROOT / "output" / "stage1" / "synthetic_benchmark.json",
    )
    causal_verification = verify_causal_invariants()
    causal_checks_passed = _causal_checks_passed(causal_verification)
    reproducible_run_available = bool(
        all(path.exists() for path in reproducibility_files)
        and catalog.get("ruleset") == examples.get("ruleset") == RULESET_VERSION
        and catalog.get("episodes")
        == sum(len(record.get("detector_results", ())) for record in catalog["records"])
        and tuning.get("inputs", {}).get("simulation_version")
        == holdout.get("inputs", {}).get("simulation_version")
        and tuning.get("inputs", {}).get("ruleset_version")
        == holdout.get("inputs", {}).get("ruleset_version")
        == RULESET_VERSION
        and tuning_checks.get("executed") == tuning_checks.get("total") == 9
        and holdout_checks.get("executed") == holdout_checks.get("total") == 9
    )
    temporal_protocol_frozen = bool(
        tuning.get("sensitivity_plan", {}).get("complete")
        and holdout.get("sensitivity_plan", {}).get("complete")
        and (ROOT / "docs" / "stage1-evaluation-protocol.md").exists()
    )
    evidence = ReadinessEvidence(
        types_covered=catalog["types_present"],
        groups_covered=examples["summary"]["reporting_groups"],
        synthetic_scenarios_covered=scenario_count,
        real_examples_reviewed=examples["summary"]["examples"],
        causal_checks_passed=causal_checks_passed,
        reproducible_run_available=reproducible_run_available,
        temporal_protocol_frozen=temporal_protocol_frozen,
        full_history_candidate_catalog=False,
        candidate_diversity_sufficient=False,
        candidates_traceable=all(examples["validation"].values()),
        external_validation_available=False,
    )
    decision = assess_readiness(evidence)
    integrity_invariants = {
        "all_19_types_in_real_sample": catalog["types_present"] == 19
        and not catalog["types_missing"],
        "all_7_groups_in_review": examples["summary"]["reporting_groups"] == 7,
        "real_examples_between_30_and_50": 30 <= examples["summary"]["examples"] <= 50,
        "all_9_scenarios_executed": tuning_checks.get("executed")
        == tuning_checks.get("total")
        == holdout_checks.get("executed")
        == holdout_checks.get("total")
        == 9,
        "tuning_sensitivity_complete": tuning["sensitivity_plan"]["complete"],
        "holdout_sensitivity_complete": holdout["sensitivity_plan"]["complete"],
        "missingness_reported_unknown": holdout_metrics["coverage"]["counts"]["unknown"] == 1,
        "isolation_forest_not_overclaimed": (
            holdout["method_comparisons"]["isolation_forest"]["status"] == "not_comparable"
        ),
        "forecast_not_declared_ready": not decision.research_forecast_ready,
        "operational_claim_not_declared_ready": not decision.operational_claim_ready,
    }
    if not all(integrity_invariants.values()):
        raise ValueError(f"Stage-1 completion integrity invariants failed: {integrity_invariants}")
    invariants = {
        **integrity_invariants,
        "all_scenario_expectations_passed": bool(tuning_checks.get("all_passed"))
        and bool(holdout_checks.get("all_passed")),
        "stage_method_complete": decision.stage_complete,
    }
    return {
        "status": (
            "stage1_method_complete_next_stage_needs_work"
            if decision.stage_complete
            else "stage1_method_incomplete_review_blockers"
        ),
        "invariants": invariants,
        "causal_verification": causal_verification,
        "real_sample": {
            "channels": catalog["channels"],
            "types": catalog["types_present"],
            "decisions": catalog["decisions"],
            "examples_reviewed": examples["summary"]["examples"],
            "groups": examples["summary"]["reporting_groups"],
            "unknown_reasons": examples["summary"]["unknown_detector_reasons"],
            "sample_only": catalog["sample_only"],
            "cadence_confirmed": catalog["cadence_confirmed"],
        },
        "synthetic": {
            "scenarios_passed_in_both_suites": scenario_count,
            "tuning_scenario_checks": tuning_checks,
            "holdout_scenario_checks": holdout_checks,
            "tuning_baseline": {
                key: tuning_metrics[key]
                for key in (
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                    "precision",
                    "recall",
                    "f1",
                )
            },
            "holdout_baseline": {
                key: holdout_metrics[key]
                for key in (
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                    "precision",
                    "recall",
                    "f1",
                    "delay_seconds_median",
                    "delay_seconds_p90",
                )
            },
            "interpretation": (
                "The benchmark now evaluates the same multi-episode pipeline as the real "
                "catalog. Holdout is a regression set already inspected during review, not a "
                "fresh independent estimate; a new frozen set is required after further tuning."
            ),
        },
        "readiness": {
            "stage_complete": decision.stage_complete,
            "research_forecast_ready": decision.research_forecast_ready,
            "operational_claim_ready": decision.operational_claim_ready,
            "completed_checks": list(decision.completed_checks),
            "blockers": list(decision.blockers),
        },
    }


def render(result: dict[str, Any]) -> str:
    real = result["real_sample"]
    synthetic = result["synthetic"]
    tuning = synthetic["tuning_baseline"]
    holdout = synthetic["holdout_baseline"]
    blockers = [_render_blocker(blocker) for blocker in result["readiness"]["blockers"]]
    blocker_text = "; ".join(blockers) if blockers else "неуказанные проверки"
    decision_text = (
        "**Методический этап завершён. Переход к обучению реального прогноза пока требует доработки.**"
        if result["readiness"]["stage_complete"]
        else f"**Методический этап ещё не закрыт: {blocker_text}.**"
    )
    return f"""# Итог первого этапа и первой недели

Дата завершения: 20 сентября 2026 года.

## Решение

{decision_text} Эксплуатационная готовность не подтверждена.

Это означает, что воспроизводимый процесс от исходной строки до объяснимого Episode,
симуляции, оценщик и отчёт существуют и проверены. Но полного исторического каталога
кандидатов ещё нет, реальная разметка отсутствует, режим регистрации и связь каналов
с объектами не подтверждены, а синтетическая устойчивость недостаточна.

## Что выполнено за Дни 1–7

1. Составлены паспорта 19 типов и матрица трёх способов обработки.
2. Проведён аудит источников, схем и производительности.
3. Реализована потоковая нормализация с сохранением источника, номера строки,
   конфликтов и дубликатов.
4. Реализованы причинная наблюдаемость, замороженные профили, объяснимые числовые,
   дискретные и контекстные правила, а также сборка эпизодов.
5. До оценки зафиксированы временные части, matching, девять сценариев и параметры
   tuning/holdout.
6. Реализованы оценщик, sensitivity, ablation и безопасный статус IsolationForest.
7. Разобраны {real["examples_reviewed"]} реальных технических примеров с охватом
   {real["types"]} типов и {real["groups"]} групп; подготовлена инструкция повторного запуска.
8. Зафиксирован контракт будущей 24-часовой метки: уже идущий эпизод не становится
   успешным прогнозом, неизвестное будущее и конец выгрузки цензурируются.

## Реальная адресная выборка

- Каналов: **{real["channels"]}**, представлены все **{real["types"]} типов**.
- Решения: {json.dumps(real["decisions"], ensure_ascii=False)}.
- Это ограниченная техническая выборка, а не полный исторический каталог.
- Частота регистрации не подтверждена; контекст объекта отсутствует.
- `candidate` не является доказанной физической поломкой, а `no_candidate` не
  подтверждает исправность.

## Синтетическая проверка

| Набор | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Tuning | {tuning["true_positives"]} | {tuning["false_positives"]} | {tuning["false_negatives"]} | {tuning["precision"]:.3f} | {tuning["recall"]:.3f} | {tuning["f1"]:.3f} |
| Holdout | {holdout["true_positives"]} | {holdout["false_positives"]} | {holdout["false_negatives"]} | {holdout["precision"]:.3f} | {holdout["recall"]:.3f} | {holdout["f1"]:.3f} |

На holdout рабочий multi-episode pipeline обнаружил {holdout["true_positives"]}
положительных сценариев, но сформировал {holdout["false_positives"]} дополнительных
предупреждений; медианная задержка — {holdout["delay_seconds_median"] / 3600:.2f} ч,
p90 — {holdout["delay_seconds_p90"] / 3600:.2f} ч. На tuning F1 равен
{tuning["f1"]:.3f}, на holdout — {holdout["f1"]:.3f}. Это подтверждает, что текущие
правила ещё требуют настройки. Holdout уже просмотрен в ходе ревью и теперь служит
регрессионным набором, а не независимой итоговой оценкой.

Малые варианты порогов и отключение numeric/discrete/context рассчитаны на одинаковом
протоколе. IsolationForest не сравнивался численно: отсутствует заранее замороженный
причинный запуск с той же экспозицией. Это зафиксировано как `not_comparable`, а не как
проигрыш или улучшение модели.

## Почему следующий этап пока не готов

{chr(10).join(f"- {blocker}" for blocker in blockers)}

## Что делать дальше

1. Разобрать {tuning["false_positives"]} tuning-FP по причинам и каналам; проверить причинный rolling-профиль,
   пороги повторного предупреждения и устойчивость без доступа к holdout.
2. Настраивать правила только на tuning; после фиксации следующей версии создать
   новый заранее отложенный набор, поскольку текущий holdout уже просмотрен.
3. Потоково построить полный каталог минимум за 2024–2026 годы и измерить покрытие,
   частоту предупреждений и разнообразие кандидатов по всем типам.
4. Получить справочник связи каналов с объектами и подтверждение режима регистрации.
5. Провести внешнюю проверку примеров. Только после этого формировать прогнозные
   пары «признаки до t → новый эпизод после t».

## Основные артефакты

- `docs/type-passports.md` — паспорта типов.
- `docs/stage1-evaluation-protocol.md` — замороженный протокол.
- `docs/stage1-simulation.md` — синтетические сценарии.
- `docs/stage1-real-examples.md` — 38 реальных разборов.
- `docs/stage1-reproducibility.md` — повтор запуска.
- `output/stage1/tuning_benchmark.json` и `synthetic_benchmark.json` — полные метрики.
- `output/stage1/completion_report.json` — машиночитаемое решение готовности.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    args = parser.parse_args()
    result = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.document.parent.mkdir(parents=True, exist_ok=True)
    args.document.write_text(render(result), encoding="utf-8")
    print(
        json.dumps(
            {"status": result["status"], "readiness": result["readiness"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
