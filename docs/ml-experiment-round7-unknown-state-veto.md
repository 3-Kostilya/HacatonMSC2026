# ML experiment round 7: smoke unknown-state veto

Research branch only. This does not change production R1/B3/Q2/R6 decisions and requires joint B/operations approval before any rollout. The 2026 test and excluded 2021 data were not read.

## Result

The round-6 second-stage specialist remains active. One additional gate applies only to a **standard 24-hour smoke warning opportunity**: suppress it if the causal `unknown_state_count_168h` feature is greater than 21. A recovery-reset warning is not vetoed by this rule. The feature comes from verified Q2/Q3 model-feature artifacts; the simulator joins it by channel, prediction time and sensor type before warning decisions. Unknown warning outcomes are not treated as negatives.

| Open development year | Candidate | Warnings | Known episodes found | Precision lower bound | Recall lower bound |
| --- | --- | ---: | ---: | ---: | ---: |
| 2024 | Round 6 | 2,287 | 399 | 17.45% | 33.14% |
| 2024 | Round 7 | 2,269 | 399 | 17.58% | 33.14% |
| 2025 | Round 6 | 3,803 | 1,120 | 29.45% | 52.29% |
| 2025 | Round 7 | 3,758 | 1,120 | 29.80% | 52.29% |

The independent audit found no lost known episode ID. In 2024 the 18 removed warnings comprised six known negatives and 12 unknown outcomes. In 2025, 46 original warning keys disappeared (18 known negatives, 28 unknown), while one additional known-negative warning arose through changed cooldown state, for a net reduction of 45. The new warning did not add or lose a known episode. The 2025 precision improvement is 0.35 percentage points on the conservative lower-bound measure, not a claim that all unknown outcomes are false alarms.

## Search and selection

`analysis/ml_experiment_round7_rule_screen.py` generated single-feature thresholds outside the observed known-positive range of 2024 standard smoke and phase warnings. It froze 17 eligible rules from 2024 before checking 2025. Many seemingly stronger rules failed transfer: the largest 2024 smoke rule (`maximum_gap_seconds_24h`) would remove 40 known-positive warnings in 2025. The chosen unknown-state rule removed no known-positive warning in the static 2024 or open-2025 screens and was then tested in a full sequential simulator, including cooldown feedback. It was chosen after seeing the open 2025 transfer result; therefore **2025 is development data, not an independent test of this round-7 choice**. A still-sealed 2026 evaluation and operational review are needed to establish generalization.

`analysis/ml_experiment_round7_pretrained_veto.py` also evaluated existing pre-2024 sensor-type specialist models with thresholds selected to preserve 2024 known hits. Their selected smoke and phase vetoes each lost three known-positive 2025 warning events, so they were rejected. Simple recovery-reset timing and longer cooldown screens also lost known 2025 hits and were not included.

## Reproduction and checks

- Frozen 2024 rule screen and open-2025 transfer: `output/ml-experiment-round7/rule-screen-v1/`.
- Full sequential results: `output/ml-experiment-round7/fullstream-2024-unknown21-v1/` and `output/ml-experiment-round7/fullstream-2025-unknown21-v2/`.
- Independent parity audit: `output/ml-experiment-round7/independent-audit-v2/report.json` (`independent_round7_warning_and_feature_parity_passed`). It rechecks feature artifact hashes, warning uniqueness, the feature threshold on every emitted standard smoke warning, and exact known episode sets.
- `python -m unittest discover -s analysis -p test_ml_experiment*.py -q`: 46 passing tests. Ruff passed on changed Python files.

Local `output/` artifacts are ignored by Git and must be regenerated for another machine; source and methodology are committed. The candidate remains experimental. In particular, do not treat the 2024/2025 no-loss observation as a guarantee for future episodes.
