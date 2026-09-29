# Технический разбор реальных примеров Stage 1

Статус: воспроизводимый технический разбор ограниченной выборки. Это **не экспертная разметка**, не подтверждение физической поломки и не оценка медицинской/промышленной безопасности. `candidate` означает только срабатывание правила `stage1-baseline-v3`; `no_candidate` не доказывает исправность.

## Покрытие

Разобраны все **38 каналов** текущего каталога: **19 типов**, **7 согласованных групп**. Решения: `candidate` — 7, `no_candidate` — 19, `unknown` — 12. Выбор примеров после просмотра результатов не выполнялся.
Всего сохранено **38 результатов-эпизодов**; их решения: `candidate` — 7, `no_candidate` — 19, `unknown` — 12.

Во всех примерах контекст объекта недоступен: в локальном справочнике нет подтверждённой связи `канал → объект`. Ожидаемая частота регистрации также не подтверждена, поэтому статус наблюдаемости преимущественно `unknown` и не переопределяет решение baseline-детектора.

## Все 38 примеров

### Числовые измерения среды

| ID | Тип / канал | Режим | Решение | Evidence | Observation quality | Контекст и граница | Исходные строки |
|---|---|---|---|---|---|---|---|
| EX-03 | Газовый датчик / `104040` | `numeric` | `candidate` | sustained_points=3; baseline_median=0.005; baseline_mad=0.005; threshold=0.03; direction=up | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Технический кандидат по правилу baseline; без экспертной разметки это не подтверждение физической неисправности. | `ext-journal-2026.7z:4999556`; `журнал_событий_пример.csv:4674`; `журнал_событий_пример.csv:4676` |
| EX-04 | Газовый датчик / `104043` | `numeric` | `no_candidate` | no_sustained_run_of_3 | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `журнал_событий_пример.csv:5789`; `журнал_событий_пример.csv:5799`; `журнал_событий_пример.csv:5802` |
| EX-11 | Датчик температуры / `120473` | `numeric` | `unknown` | insufficient_numeric_history:6<15 | unknown: insufficient_numeric_history:6<15 | объект `unknown`; Решение намеренно не принято из-за качества или недостатка истории; состояние оборудования неизвестно. | `ext-journal-2026.7z:4075760`; `ext-journal-2026.7z:4147844`; `ext-journal-2026.7z:4583102` |
| EX-12 | Датчик температуры / `120475` | `numeric` | `unknown` | insufficient_numeric_history:2<15 | unknown: insufficient_numeric_history:2<15 | объект `unknown`; Решение намеренно не принято из-за качества или недостатка истории; состояние оборудования неизвестно. | `ext-journal-2026.7z:4831890`; `журнал_событий_пример.csv:3` |

### Пожарные извещатели

| ID | Тип / канал | Режим | Решение | Evidence | Observation quality | Контекст и граница | Исходные строки |
|---|---|---|---|---|---|---|---|
| EX-07 | Датчик дыма / `196487` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:3257953`; `ext-journal-2026.7z:3639791`; `ext-journal-2026.7z:3641669` |
| EX-08 | Датчик дыма / `196976` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:4526082`; `ext-journal-2026.7z:4526184`; `ext-journal-2026.7z:4527378` |
| EX-37 | Тепловой датчик / `93679` | `discrete` | `candidate` | transitions=4; window_seconds=600 | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Технический кандидат по правилу baseline; без экспертной разметки это не подтверждение физической неисправности. | `ext-journal-2026.7z:4699922`; `ext-journal-2026.7z:4701159`; `ext-journal-2026.7z:4701808` |
| EX-38 | Тепловой датчик / `93680` | `discrete` | `candidate` | transitions=4; window_seconds=600 | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Технический кандидат по правилу baseline; без экспертной разметки это не подтверждение физической неисправности. | `ext-journal-2026.7z:4699923`; `ext-journal-2026.7z:4701160`; `ext-journal-2026.7z:4701814` |

### Охранные и контактные каналы

| ID | Тип / канал | Режим | Решение | Evidence | Observation quality | Контекст и граница | Исходные строки |
|---|---|---|---|---|---|---|---|
| EX-01 | 9-секционный люк / `330498` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:2610545`; `ext-journal-2026.7z:3778359`; `ext-journal-2026.7z:3778385` |
| EX-02 | 9-секционный люк / `330500` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:2610547`; `ext-journal-2026.7z:3778361`; `ext-journal-2026.7z:3778387` |
| EX-05 | Датчик движения / `266605` | `discrete` | `unknown` | blocking_quality:channel_time_conflict | unknown: blocking_quality:channel_time_conflict | объект `unknown`; Решение намеренно не принято из-за качества или недостатка истории; состояние оборудования неизвестно. | `ext-journal-2026.7z:2902271`; `ext-journal-2026.7z:2902283`; `ext-journal-2026.7z:2902285` |
| EX-06 | Датчик движения / `266620` | `discrete` | `candidate` | transitions=4; window_seconds=600 | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Технический кандидат по правилу baseline; без экспертной разметки это не подтверждение физической неисправности. | `ext-journal-2026.7z:2885377`; `ext-journal-2026.7z:2885385`; `ext-journal-2026.7z:2885396` |
| EX-15 | КД АВ / `266604` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:2902384`; `ext-journal-2026.7z:4902549`; `ext-journal-2026.7z:4902829` |
| EX-16 | КД АВ / `266609` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:2902385`; `ext-journal-2026.7z:4902554`; `ext-journal-2026.7z:4902834` |
| EX-17 | КД Дверь / `266607` | `discrete` | `unknown` | blocking_quality:channel_time_conflict | unknown: blocking_quality:channel_time_conflict | объект `unknown`; Решение намеренно не принято из-за качества или недостатка истории; состояние оборудования неизвестно. | `ext-journal-2026.7z:1206563`; `ext-journal-2026.7z:1206577`; `ext-journal-2026.7z:1206578` |
| EX-18 | КД Дверь / `266608` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:4019272`; `ext-journal-2026.7z:4902553`; `ext-journal-2026.7z:4902833` |
| EX-19 | КД Люк / `277972` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:3866402`; `ext-journal-2026.7z:3965556`; `ext-journal-2026.7z:3966654` |
| EX-20 | КД Люк / `277973` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:3866403`; `ext-journal-2026.7z:3965557`; `ext-journal-2026.7z:3966655` |
| EX-35 | Стекло / `77539` | `discrete` | `candidate` | transitions=4; window_seconds=600 | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Технический кандидат по правилу baseline; без экспертной разметки это не подтверждение физической неисправности. | `ext-journal-2026.7z:2010747`; `ext-journal-2026.7z:3063076`; `ext-journal-2026.7z:3063088` |
| EX-36 | Стекло / `77542` | `discrete` | `candidate` | transitions=4; window_seconds=600 | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Технический кандидат по правилу baseline; без экспертной разметки это не подтверждение физической неисправности. | `ext-journal-2026.7z:2010748`; `ext-journal-2026.7z:3063243`; `ext-journal-2026.7z:3063256` |

### Контроль затопления

| ID | Тип / канал | Режим | Решение | Evidence | Observation quality | Контекст и граница | Исходные строки |
|---|---|---|---|---|---|---|---|
| EX-09 | Датчик затопления / `266705` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:4902868`; `ext-journal-2026.7z:4972742`; `ext-journal-2026.7z:4973570` |
| EX-10 | Датчик затопления / `266839` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:4902929`; `ext-journal-2026.7z:4972903`; `ext-journal-2026.7z:4973644` |

### Состояние оборудования

| ID | Тип / канал | Режим | Решение | Evidence | Observation quality | Контекст и граница | Исходные строки |
|---|---|---|---|---|---|---|---|
| EX-27 | Состояние вентилятора / `115627` | `discrete` | `unknown` | blocking_quality:channel_time_conflict | unknown: blocking_quality:channel_time_conflict | объект `unknown`; Решение намеренно не принято из-за качества или недостатка истории; состояние оборудования неизвестно. | `ext-journal-2026.7z:3181401`; `ext-journal-2026.7z:4110070`; `ext-journal-2026.7z:4110071` |
| EX-28 | Состояние вентилятора / `95453` | `discrete` | `candidate` | transitions=4; window_seconds=600 | unknown: insufficient_history; cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Технический кандидат по правилу baseline; без экспертной разметки это не подтверждение физической неисправности. | `журнал_событий_пример.csv:6206`; `журнал_событий_пример.csv:6207`; `журнал_событий_пример.csv:6208` |
| EX-29 | Состояние насоса / `103937` | `discrete` | `unknown` | blocking_quality:channel_time_conflict | unknown: blocking_quality:channel_time_conflict | объект `unknown`; Решение намеренно не принято из-за качества или недостатка истории; состояние оборудования неизвестно. | `ext-journal-2026.7z:426093`; `ext-journal-2026.7z:453552`; `ext-journal-2026.7z:453553` |
| EX-30 | Состояние насоса / `103979` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:37285`; `ext-journal-2026.7z:58063`; `ext-journal-2026.7z:58211` |

### Питание и состояние аппаратуры

| ID | Тип / канал | Режим | Решение | Evidence | Observation quality | Контекст и граница | Исходные строки |
|---|---|---|---|---|---|---|---|
| EX-13 | ИБП / `162867` | `discrete` | `unknown` | blocking_quality:channel_time_conflict | unknown: blocking_quality:channel_time_conflict | объект `unknown`; Решение намеренно не принято из-за качества или недостатка истории; состояние оборудования неизвестно. | `ext-journal-2026.7z:1822788`; `ext-journal-2026.7z:2465022`; `ext-journal-2026.7z:2504821` |
| EX-14 | ИБП / `50973` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:2464163`; `ext-journal-2026.7z:3057184`; `ext-journal-2026.7z:3057188` |
| EX-25 | Состояние УИР-Р / `229736` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:2565847`; `ext-journal-2026.7z:4248148`; `ext-journal-2026.7z:4248583` |
| EX-26 | Состояние УИР-Р / `230192` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:2565848`; `ext-journal-2026.7z:4248150`; `ext-journal-2026.7z:4248584` |
| EX-33 | Состояние фазы / `103945` | `discrete` | `unknown` | blocking_quality:channel_time_conflict | unknown: blocking_quality:channel_time_conflict | объект `unknown`; Решение намеренно не принято из-за качества или недостатка истории; состояние оборудования неизвестно. | `ext-journal-2026.7z:3776649`; `ext-journal-2026.7z:3818674`; `ext-journal-2026.7z:3818698` |
| EX-34 | Состояние фазы / `104014` | `discrete` | `unknown` | blocking_quality:channel_time_conflict | unknown: blocking_quality:channel_time_conflict | объект `unknown`; Решение намеренно не принято из-за качества или недостатка истории; состояние оборудования неизвестно. | `ext-journal-2026.7z:3754816`; `ext-journal-2026.7z:3818690`; `ext-journal-2026.7z:3818693` |

### Управление, ручные сигналы и режим охраны

| ID | Тип / канал | Режим | Решение | Evidence | Observation quality | Контекст и граница | Исходные строки |
|---|---|---|---|---|---|---|---|
| EX-21 | Переключатель / `104011` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:832420`; `ext-journal-2026.7z:1018469`; `ext-journal-2026.7z:1053414` |
| EX-22 | Переключатель / `104018` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:832395`; `ext-journal-2026.7z:1018464`; `ext-journal-2026.7z:1053402` |
| EX-23 | Ручной извещатель / `230701` | `discrete` | `no_candidate` | no_sustained_discrete_pattern | unknown: cadence_unknown; one_or_more_windows_unusable | объект `unknown`; Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности. | `ext-journal-2026.7z:2565851`; `ext-journal-2026.7z:4248156`; `ext-journal-2026.7z:4248587` |
| EX-24 | Ручной извещатель / `334526` | `discrete` | `unknown` | insufficient_discrete_history:6<7 | unknown: insufficient_discrete_history:6<7 | объект `unknown`; Решение намеренно не принято из-за качества или недостатка истории; состояние оборудования неизвестно. | `ext-journal-2026.7z:1924463`; `ext-journal-2026.7z:1926488`; `ext-journal-2026.7z:2466546` |
| EX-31 | Состояние охраны / `267934` | `discrete` | `unknown` | blocking_quality:channel_time_conflict | exclude: blocking_quality:channel_time_conflict | объект `unknown`; Решение намеренно не принято из-за качества или недостатка истории; состояние оборудования неизвестно. | `ext-journal-2026.7z:1151511`; `ext-journal-2026.7z:1151513`; `ext-journal-2026.7z:1222386` |
| EX-32 | Состояние охраны / `9330` | `discrete` | `unknown` | blocking_quality:channel_time_conflict | exclude: blocking_quality:channel_time_conflict | объект `unknown`; Решение намеренно не принято из-за качества или недостатка истории; состояние оборудования неизвестно. | `ext-journal-2026.7z:234914`; `ext-journal-2026.7z:234915`; `ext-journal-2026.7z:237426` |

## Почему получен `unknown`

- `blocking_quality:channel_time_conflict` — 9 канал(а/ов).
- `insufficient_discrete_history:6<7` — 1 канал(а/ов).
- `insufficient_numeric_history:2<15` — 1 канал(а/ов).
- `insufficient_numeric_history:6<15` — 1 канал(а/ов).

`blocking_quality:channel_time_conflict` означает, что у канала в одну секунду сохранены разные значения: пайплайн не выбирает одно из них молча. `insufficient_*_history` означает, что до точки проверки нет минимального числа пригодных наблюдений. Это причины воздержаться от решения, а не признаки неисправности.

## Три сквозных маршрута

### EX-03: `candidate`

1. Исходная ссылка: `журнал_событий_пример.csv:4674`, событие `4524059002`, время `2026-08-01 00:00:25`, исходное значение `0.07`, флаги качества `[]`, роль строки `episode_interval`.
2. Нормализация сохранила исходный текст и отдельно числовое значение; режим обработки — `numeric`.
3. Детектор `numeric_level_shift` сформировал основания: `sustained_points=3; baseline_median=0.005; baseline_mad=0.005; threshold=0.03; direction=up`.
4. Получен Episode `ep-7f4dfb615693f782`: начало `2026-08-01 00:00:25`, подтверждение `2026-08-01 00:02:30`, решение `candidate`.
5. Ограничение: Технический кандидат по правилу baseline; без экспертной разметки это не подтверждение физической неисправности.

### EX-01: `no_candidate`

1. Исходная ссылка: `ext-journal-2026.7z:3778359`, событие `3296920009`, время `2026-01-27 10:42:50`, исходное значение `Неопределен`, флаги качества `[]`, роль строки `episode_interval`.
2. Нормализация сохранила исходный текст и отдельно числовое значение; режим обработки — `discrete`.
3. Детектор `discrete_pattern` сформировал основания: `no_sustained_discrete_pattern`.
4. Получен Episode `ep-9826ab2e91ef2fd1`: начало `2026-01-27 10:42:50`, подтверждение `2026-01-29 15:53:40`, решение `no_candidate`.
5. Ограничение: Правило не нашло паттерн в ограниченной выборке; это не подтверждение исправности.

### EX-05: `unknown`

1. Исходная ссылка: `ext-journal-2026.7z:2902283`, событие `3292365010`, время `2026-01-21 12:07:50`, исходное значение `Движения нет`, флаги качества `['channel_time_conflict']`, роль строки `episode_interval`.
2. Нормализация сохранила исходный текст и отдельно числовое значение; режим обработки — `discrete`.
3. Детектор `discrete_pattern` сформировал основания: `blocking_quality:channel_time_conflict`.
4. Получен Episode `ep-233750bdda71632e`: начало `2026-01-21 12:07:50`, подтверждение `2026-01-21 12:07:50`, решение `unknown`.
5. Ограничение: Решение намеренно не принято из-за качества или недостатка истории; состояние оборудования неизвестно.

## Машиночитаемая детализация

Полные записи находятся в `output/stage1/real_examples.json`. Для каждого примера там сохранены до шести ссылок `source + source_row`, исходные значения, доступный контекст, сводка канала, evidence, качество наблюдаемости, полный Episode и граница интерпретации. Команды полного повторного запуска и контрольные суммы описаны в `docs/stage1-reproducibility.md`.
