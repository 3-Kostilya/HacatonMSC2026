# Воспроизводимость Stage 1: Дни 5–7

Документ описывает повтор текущего ограниченного прохода от исходных журналов до
каталога Episode и технического разбора 38 каналов. Результат относится только к
зафиксированной выборке и правилам `stage1-baseline-v3`. Он не является экспертной
разметкой физических неисправностей.

## Зафиксированное окружение

- Windows, PowerShell;
- Python 3.14.3;
- PyArrow 25.0.1;
- зависимости из `requirements.txt`;
- Git HEAD на момент разбора: `0082cf9 (исходный коммит; изменения v3 ещё не закоммичены)`;
- системный 7-Zip, который обнаруживает `analysis.audit_stage1_sources.find_seven_zip`.

Рабочее дерево на момент выполнения содержало незакоммиченные файлы, поэтому Git
HEAD сам по себе не определяет результат. Истиной для данного прогона служат также
контрольные суммы двух непосредственных входов, записанные в
`output/stage1/real_examples.json`.

## Входы и границы

Первый шаг читает:

- `data/журнал_событий_пример.csv`;
- первые не более 5 000 000 строк `data/ext-journal-2026.7z`;
- `data/справочник_каналов_датчиков.csv` для типа канала;
- не более двух первых наблюдавшихся каналов каждого типа; для редкого
  `9-секционный люк` IDs предварительно берутся из справочника.

Это намеренно ограниченная выборка, а не полный расчёт истории. Физическая строка
источника задаётся парой `source + source_row`; номер однобазовый и учитывает строку
заголовка CSV. Архив 2021 года локально отсутствует и в этот проход не входит.

Непосредственные входы разбора при проверенном запуске:

- `normalized_sample.parquet` — SHA-256
  `e3f6023c24799a8f2341cb3975c1264b7dd01db18f49c564c36fd8d07660d46e`;
- `baseline_catalog.json` — SHA-256
  `20e81274037144461c29c9b27b4912a9439344724b1a39bbc44c54bad6e5d03e`.

Если предыдущие шаги пересозданы, их суммы могут обоснованно измениться. В таком
случае сравнивать нужно покрытие, проверки и содержимое Episode, а не ожидать
совпадения с прежними суммами.

## Полный повторный запуск

Команды выполняются из корня проекта в активированной `.venv`:

```powershell
python analysis/prepare_stage1_sample.py `
  --input "data/журнал_событий_пример.csv" `
  --input "data/ext-journal-2026.7z" `
  --max-input-rows 5000000 `
  --channels-per-type 2 `
  --output "output/stage1/normalized_sample.parquet"

python analysis/run_stage1_baseline.py `
  --input "output/stage1/normalized_sample.parquet" `
  --output "output/stage1/baseline_catalog.json"

python analysis/build_stage1_examples.py `
  --catalog "output/stage1/baseline_catalog.json" `
  --events "output/stage1/normalized_sample.parquet" `
  --output "output/stage1/real_examples.json" `
  --report "docs/stage1-real-examples.md"

python analysis/run_synthetic_benchmark.py `
  --events "output/stage1/tuning_events.jsonl" `
  --truth "output/stage1/tuning_truth_manifest.json" `
  --output "output/stage1/tuning_benchmark.json"

python analysis/run_synthetic_benchmark.py
python analysis/build_stage1_final_report.py

python -X utf8 -m unittest discover -s stage1 -p "test_*.py" -v
python -X utf8 -m unittest discover -s analysis -p "test_*.py" -v
```

Последний шаг должен напечатать: 38 примеров, 19 типов, 7 групп, решения
`candidate=7`, `no_candidate=19`, `unknown=12` — после исправлений ревью и при
неизменных входах.
Режим проверки без записи файлов:

```powershell
python analysis/build_stage1_examples.py --check
```

Встроенная проверка аварийно завершает запуск, если примеров не 30–50, нет хотя бы
одного из 19 типов или семи групп, отсутствует один из трёх исходов, у примера нет
ссылки на строку либо у `unknown` нет причины.

## Выходы

| Файл | Содержание |
|---|---|
| `output/stage1/normalized_sample.parquet` | Нормализованные события с сохранённым `raw_value`, источником, строкой и флагами качества |
| `output/stage1/normalized_sample.manifest.json` | Входы, лимиты и покрытие выборки |
| `output/stage1/baseline_catalog.json` | Один объяснимый Episode на канал текущего baseline-прохода |
| `output/stage1/real_examples.json` | 38 полных технических карточек, контрольные суммы и автоматические проверки покрытия |
| `docs/stage1-real-examples.md` | Читаемый разбор всех карточек и три сквозных маршрута |

## Маршрут от строки до Episode

1. `prepare_stage1_sample.py` читает строку и передаёт её в общий нормализатор.
   Исходный текст значения не заменяется числом; число хранится отдельно.
2. Нормализатор сохраняет `source`, `source_row`, `event_id`, локальное время,
   `raw_value`, `numeric_value`, `alarm` и флаги качества. Разные значения одного
   канала в одну секунду не схлопываются, а получают
   `channel_time_conflict`.
3. `run_stage1_baseline.py` группирует принятые события по типу и каналу. Реестр
   выбирает числовой или дискретный детектор; профиль строится только по доступной
   истории до проверяемой точки.
4. Детектор создаёт Episode с `start_at`, `confirmed_at`, `decision`, `evidence`,
   `observation_quality` и версией правил. При блокирующем конфликте или недостатке
   истории выдаётся `unknown`.
5. `build_stage1_examples.py` связывает Episode обратно с исходными строками,
   добавляет сводку канала и явное ограничение интерпретации. Он не пересматривает
   решение детектора и не присваивает экспертную метку.

Конкретные маршруты для `candidate`, `no_candidate` и `unknown` приведены в
`docs/stage1-real-examples.md`; полный набор связанных строк находится в JSON.

## Проверка и известные ограничения

Проверенный запуск:

```powershell
python analysis/build_stage1_examples.py --check
ruff check analysis/build_stage1_examples.py
```

Обязательные проверки покрытия прошли. Отдельный тестовый модуль не добавлялся:
builder содержит проверку инвариантов на фактических двух входах, а создание файлов
в этой работе было намеренно ограничено builder, двумя документами и его JSON.

Ограничения результата:

- нет экспертных меток, поэтому нельзя вычислять фактические precision/recall;
- `candidate` — техническое срабатывание, а не доказанная поломка;
- `no_candidate` — отсутствие паттерна в выборке, а не доказанная исправность;
- ожидаемая частота регистрации не подтверждена;
- нет подтверждённой связи `канал → объект`, поэтому объектный контекст `unknown`;
- смысл и единицы отдельных типов, включая газ, ИБП, КД АВ и УИР-Р, не должны
  домысливаться;
- выборка ограничена двумя каналами на тип и не доказывает переносимость на все
  11 485 каналов или другие годы.
