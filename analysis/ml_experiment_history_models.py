"""Run pooled learners with causal, channel-specific long-term recurrence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.ml_experiment_features import engineered_input
from analysis.ml_experiment_history import CompletedHistory
from analysis.ml_experiment_pooled import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('output/ml-experiment/data'))
    parser.add_argument('--output', type=Path, default=Path('output/ml-experiment/history'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    history = CompletedHistory.load()

    def feature_builder(frame, base_names, *, engineered):
        matrix = engineered_input(frame, base_names, engineered=engineered)
        past = history.features(frame).fillna(-1).astype('float32')
        return pd.concat([matrix, past], axis=1)

    configs = {
        'history_moderate': {'engineered': True, 'weights': 'sqrt', 'iterations': 300, 'depth': 5},
        'history_episode': {'engineered': True, 'weights': 'episode', 'iterations': 300, 'depth': 5},
    }
    report = run(args.data, args.output, feature_builder=feature_builder, configs_override=configs)
    report['history_contract'] = {
        'source': 'accepted B2 registered episodes with exact unambiguous Norma closure',
        'availability': 'end_at <= prediction_time',
        'excluded_segment': 2021,
        'windows_days': [7, 28, 90, 365],
        'identity_columns_used_as_predictors': False,
        'closure_2026_or_later_included': False,
    }
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
