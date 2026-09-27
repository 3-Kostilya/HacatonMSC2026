"""One final linear refit through 2024 with previously frozen configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.base import clone

from analysis.ml_experiment_eval import PreparedEvaluation
from analysis.ml_experiment_linear import score_file, transform
from analysis.prepare_ml_experiment import sha256


def run(root: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    data = root / 'data'
    folder = root / 'linear'
    selection = json.loads((folder / 'frozen_selection_canonical_v2.json').read_text(encoding='utf-8'))
    name = selection['selected_variant']
    spec = next(item for item in selection['variants'] if item['name'] == name)
    threshold = selection['selections'][name]['threshold']
    output.mkdir(parents=True)
    # Capture the chosen hyperparameters and threshold before the refit.
    frozen = {'selected_variant': name, 'threshold': threshold, 'variant_spec': spec,
              'source_selection_sha256': sha256(folder / 'frozen_selection_canonical_v2.json'),
              'fit_end_exclusive': '2025-01-01', 'selection_year': 2024,
              'score_scale_assumption': 'original 2024 threshold retained after final refit'}
    (output / 'selection.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding='utf-8')
    old_model = joblib.load(folder / f'{name}.joblib')
    model = clone(old_model)
    train = pq.ParquetFile(data / 'refit_train.parquet').read().to_pandas()
    y = train.target.to_numpy(dtype='int8')
    positive = y == 1
    weights = np.ones(len(y), dtype='float64')
    if name == 'log_episode_sqrt':
        counts = train.loc[positive, 'target_episode_id'].value_counts()
        weights[positive] = 1 / train.loc[positive, 'target_episode_id'].map(counts).to_numpy()
        weights[positive] *= positive.sum() / weights[positive].sum()
        model.set_params(sgdclassifier__class_weight={0: 1, 1: float(np.sqrt((~positive).sum()/positive.sum()))})
    model.fit(transform(train, spec['names'], name, spec['limits']), y,
              sgdclassifier__sample_weight=weights)
    joblib.dump(model, output / f'{name}.joblib')
    del train
    score_file(data / 'validation.parquet', output / 'validation_scores.parquet', [spec], {name: model})
    frame = pq.ParquetFile(output / 'validation_scores.parquet').read().to_pandas()
    metric = PreparedEvaluation(frame, 2142).evaluate(f'score_{name}', threshold)
    report = {'selection': frozen, 'validation': metric,
              'scope': 'one final refit through 2024, unchanged threshold, already-open2025 transfer',
              'hashes': {'model': sha256(output / f'{name}.joblib'),
                         'scores': sha256(output / 'validation_scores.parquet'),
                         'refit_source_manifest': sha256(data / 'refit_manifest.json')}}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: metric[k] for k in ('episode_precision','full_episode_recall','full_episode_f1','matched_episodes','emitted_warnings')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/ml-experiment'))
    parser.add_argument('--output', type=Path, default=Path('output/ml-experiment/linear-refit'))
    args = parser.parse_args()
    run(args.root, args.output)
