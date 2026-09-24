# Исследование исходного датасета

Быстрая проверка текущих справочников по уже сохранённым частотам каналов:

```bash
python -X utf8 analysis/check_dictionary.py
```

Она не распаковывает годовые журналы: сверяет размеры архивов и метаданные вложенных CSV,
повторно читает небольшие справочники и пример журнала. Нужны готовые
`analysis/results/ext-journal-*.json`, `dictionaries.json` и `журнал_событий_пример.json`.
Результат: `output/dictionary_check/report.md`, числовой аудит и список неизвестных каналов
в JSON. Старые результаты и исходные данные не изменяются.

Сводка качества данных и сравнение сценариев по сохранённым результатам:

```bash
python -X utf8 analysis/check_data_quality.py
```

Она использует годовые профили и предыдущую проверку справочника; результат сохраняется
в `output/data_quality/report.md` и `audit.json`.

Для детального профиля температурных каналов за 2024–2026 годы:

```bash
python -X utf8 analysis/profile_temperature.py
python -X utf8 analysis/report_temperature.py
```

Первый скрипт читает годовые архивы и проверяет число отобранных и числовых записей по
сохранённым итогам. Результаты: `output/temperature_profile/channel_year.csv` и
`audit.json`. Второй собирает `report.md` без повторного чтения архивов.

Первый температурный сценарий на канале 286947 строится после выгрузки событий
кандидатных каналов:

```bash
python -X utf8 analysis/extract_temperature_cohort.py
python -X utf8 analysis/build_temperature_demo.py
```

График, предупреждения и разбор результата находятся в `output/temperature_anomaly/`.
Предварительная выгрузка одного канала 2943 воспроизводится через
`analysis/extract_temperature_channel.py`; её быстрые повторяющиеся проходы значений
сделали канал неподходящим для простого правила отклонения от среднего.

Скрипты первичного анализа всех годовых архивов, примера журнала и справочников. Это отдельный исследовательский процесс, он не нужен для обучения CatBoost.

Дополнительные зависимости:

```bash
python -m pip install -r analysis/requirements.txt
```

Порядок запуска из корня проекта:

```bash
python -X utf8 analysis/profile_dataset.py
python -X utf8 analysis/summarize_results.py
python -X utf8 analysis/compare_sample.py
python -X utf8 analysis/build_report.py
python -X utf8 analysis/verify_report.py
```

Для этих исторических скриптов требуется Windows со стандартной установкой 7-Zip и шрифтами Arial. Основной ML-процесс поддерживает поиск 7-Zip через PATH и переменную `SEVEN_ZIP`.

Исходники находятся в `data/`. Промежуточные результаты сохраняются в `analysis/results/`, PDF — в `output/pdf/`, изображения проверки — в `tmp/pdfs/`. Профилирование всего датасета сохраняет крупные массивы идентификаторов и требует нескольких гигабайт свободного места. Все сгенерированные файлы исключены из Git.
