# Передача B: сверка Q2 и точный потолок Recall

Дата: 27.09.2026. Код и документы — в `ML`, данные — только вне Git.
Основной итог: [решение A](ml-q2-a-b-acceptance-and-oracle-review.md).

## Что передать

ZIP: `output/q2-a-verification-handoff-20260927.zip`.

- Размер: **195 342 байта** (около 191 KiB), 31 проверенный файл.
- SHA-256 ZIP: `7b09dd917cc598f78bc2c07212a51928b75a3a8ca6d2fee0ee286ae0446e1186`.
- SHA-256 внутреннего манифеста:
  `953929a5ea22fe79018fc415ee8897529d1028b0350dc0d47a043a7df04a4c78`.
- Проверено чтение каждого файла из ZIP, CRC и SHA-256; временные файлы,
  сырой M1, модели и полные признаки не включены.

Распаковать в `output/`. Получится каталог
`output/q2-a-verification-handoff-20260927/` с:

1. `q2-a-oracle-capacity-20260927/`: отчёт, манифест, расписание оракула
   и ёмкость каждого канала. **Расписание использует будущую разметку;
   не использовать для признаков, допуска, обучения или живого пилота.**
2. `q2-a-b-score-verification-20260927/`: итог и 12 помесячных сверок всех баллов.
3. `q2-a-b-metrics-acceptance-20260927/`: приёмка A и независимо рассчитанные
   ключи 1 305 предупреждений.
4. Воспроизведённые файлы B: сетка порогов, окончательное решение, аудит
   эпизодов/предупреждений и газового кластера с прежними хешами B.

B уже имеет основной Q2/A и свой каталог прогнозов. Этот ZIP их не
заменяет. У ресурсоёмких запусков A отчёты содержат время/память текущей
машины; при новом запуске их собственные хеши могут измениться. Сравнивать
исходные пины, ключи, числа и результаты проверок, а не время запуска.

## Что независимо проверить B

Воспроизвести максимум **1 116** по собственным сохранённым прогнозам и
сверить расписание/поканальную ёмкость. Динамическое программирование A
и жадный расчёт дают одинаковый максимум; это потолок, не Recall модели.
Затем совместно рассмотреть [вопросы о смысле эпизодов](ml-q2-episode-alert-questions.md).
Повторное обучение для этой проверки не требуется.

## Воспроизведение из корня проекта

Новые назначения вывода ниже не должны существовать. На машине B можно
заменить префикс `check-*`, сохранив входные пакеты с указанными хешами.
В PowerShell использовать Python проекта вместо системного, например
`./.venv/Scripts/python.exe`. Следующие команды не обращаются к старому test.

```powershell
./.venv/Scripts/python.exe -m analysis.audit_q2_oracle_a --q2-dir output/q2-a-full-sparse-20260926-v5 --experiment output/q2-b-expanded-validation-20260927 --output-dir output/check-q2-oracle

./.venv/Scripts/python.exe -m analysis.verify_q2_b_scores_a --experiment output/q2-b-expanded-validation-20260927 --q2-dir output/q2-a-full-sparse-20260926-v5 --b3-dir output/r3-b-full-months-20260925-v2 --output-dir output/check-q2-scores

./.venv/Scripts/python.exe -m analysis.refine_q2_b_thresholds --experiment output/q2-b-expanded-validation-20260927 --output output/check-q2-refined
./.venv/Scripts/python.exe -m analysis.finalize_q2_b_thresholds --experiment output/q2-b-expanded-validation-20260927 --refinement output/check-q2-refined --output output/check-q2-decision.json
./.venv/Scripts/python.exe -m analysis.audit_q2_b_errors --experiment output/q2-b-expanded-validation-20260927 --decision output/check-q2-decision.json --q2-dir output/q2-a-full-sparse-20260926-v5 --output-dir output/check-q2-errors
./.venv/Scripts/python.exe -m analysis.audit_q2_b_gas_burst --error-dir output/check-q2-errors --experiment output/q2-b-expanded-validation-20260927 --decision output/check-q2-decision.json --q2-dir output/q2-a-full-sparse-20260926-v5 --m1-dir output/milestone1/full_20260922 --output output/check-q2-gas.json
./.venv/Scripts/python.exe -m analysis.verify_q2_b_metrics_a --experiment output/q2-b-expanded-validation-20260927 --q2-dir output/q2-a-full-sparse-20260926-v5 --decision output/check-q2-decision.json --error-dir output/check-q2-errors --gas-report output/check-q2-gas.json --score-verification output/check-q2-scores --output-dir output/check-q2-acceptance
```

Фиксированная сетка воспроизводится потому, что отдельные производные
файлы B отсутствовали в передаче. Итог выбора порога, манифест ошибок и
газовый отчёт на машине A совпали с опубликованными B побайтно.

## Проверки кода

```powershell
./.venv/Scripts/python.exe -m unittest discover -s analysis -p 'test*.py'
./.venv/Scripts/python.exe -m unittest discover -s stage1 -p 'test*.py'
./.venv/Scripts/python.exe -m unittest discover -s ml/forecast -p 'test*.py'
./.venv/Scripts/python.exe -m ruff check .
git diff --check
```

14 новых тестов проверяют границу ровно 24 часа, влияние ложного
предупреждения на последующее истинное, двустороннюю сверку ключей,
неизменность меток/баллов, точность оракула против полного перебора
маленьких случаев и целостность передачи.

На машине A прошли **218 аналитических тестов, 300 тестов `stage1`
и 10 тестов `ml/forecast`**, всего 528. Полная проверка стиля, форматирование
восьми новых файлов и проверка пробелов в изменениях также прошли.
