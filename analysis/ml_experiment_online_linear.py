"""Quarterly causal refits with fixed 2024 configuration and operating threshold.

This retrospective, previously opened 2025 experiment simulates learning from
already available outcomes. Every new model has label_available_at < cutoff.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.base import clone

from analysis.ml_experiment_eval import PreparedEvaluation
from analysis.ml_experiment_linear import KEYS, transform
from analysis.prepare_ml_experiment import sha256


def available_training_rows(db, past_path: Path, cutoff: str) -> pd.DataFrame:
    return db.execute("""SELECT * FROM read_parquet(?)
        WHERE label_available_at < CAST(? AS TIMESTAMP)
        AND prediction_time < CAST(? AS TIMESTAMP)
        AND (target=1 OR hash(channel_id,prediction_time)%10000<200)""",
        [str(past_path), cutoff, cutoff]).fetch_df()


def run(root: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    data = root / 'data'
    selection = json.loads((root/'linear/frozen_selection_canonical_v2.json').read_text(encoding='utf-8'))
    name = selection['selected_variant']
    spec = next(x for x in selection['variants'] if x['name'] == name)
    threshold = selection['selections'][name]['threshold']
    output.mkdir(parents=True)
    frozen = {'selected_variant': name, 'threshold': threshold,
              'configuration_source_sha256': sha256(root/'linear/frozen_selection_canonical_v2.json'),
              'selection_year': 2024, 'updates': ['2025-01-01','2025-04-01','2025-07-01','2025-10-01'],
              'label_rule': 'strictly label_available_at < quarter_start',
              'threshold_adapted_using_2025_labels': False}
    (output/'selection.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    base = pq.ParquetFile(data/'refit_train.parquet').read().to_pandas()
    original_model = joblib.load(root/'linear-refit'/f'{name}.joblib')
    writer = None
    fits = []
    boundaries = ['2025-01-01','2025-04-01','2025-07-01','2025-10-01','2026-01-01']
    with duckdb.connect(':memory:') as db:
        db.execute('SET threads=2')
        db.execute("SET memory_limit='1GB'")
        for quarter, (cutoff, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
            if quarter == 1:
                model = original_model
                train = base
            else:
                past = available_training_rows(db, data/'validation.parquet', cutoff)
                train = pd.concat([base, past], ignore_index=True)
                del past
                if train.label_available_at.max() >= pd.Timestamp(cutoff):
                    raise ValueError('quarterly fit contains an unavailable outcome')
                model = clone(original_model)
                y = train.target.to_numpy(dtype='int8')
                positive = y == 1
                weight = np.ones(len(train), dtype='float64')
                if name == 'log_episode_sqrt':
                    counts = train.loc[positive, 'target_episode_id'].value_counts()
                    weight[positive] = 1/train.loc[positive, 'target_episode_id'].map(counts).to_numpy()
                    weight[positive] *= positive.sum()/weight[positive].sum()
                    model.set_params(sgdclassifier__class_weight={0:1, 1:float(np.sqrt((~positive).sum()/positive.sum()))})
                matrix = transform(train, spec['names'], name, spec['limits'])
                model.fit(matrix, y, sgdclassifier__sample_weight=weight)
                del matrix, y, positive, weight
            joblib.dump(model, output/f'quarter_{quarter}.joblib')
            fits.append({'quarter': quarter, 'cutoff': cutoff, 'rows': len(train),
                         'positive_hours': int(train.target.sum()),
                         'maximum_label_available_at': str(train.label_available_at.max()),
                         'model_sha256': sha256(output/f'quarter_{quarter}.joblib')})
            if quarter > 1:
                del train
            gc.collect()
            query = db.execute("""SELECT * FROM read_parquet(?)
                WHERE prediction_time >= CAST(? AS TIMESTAMP)
                AND prediction_time < CAST(? AS TIMESTAMP)""", [str(data/'validation.parquet'),cutoff,end])
            reader = query.to_arrow_reader(batch_size=100_000)
            quarter_rows = 0
            for batch in reader:
                frame = batch.to_pandas()
                out = frame[KEYS].copy()
                out['score_online'] = model.predict_proba(transform(frame, spec['names'], name, spec['limits']))[:,1].astype('float32')
                out['model_quarter'] = np.full(len(out), quarter, dtype='int8')
                table = pa.Table.from_pandas(out, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output/'validation_scores.parquet', table.schema, compression='zstd')
                writer.write_table(table)
                quarter_rows += len(out)
            print('completed quarter',quarter,'fit_rows',fits[-1]['rows'],'scored_rows',quarter_rows, flush=True)
            (output/'fit_manifest.json').write_text(json.dumps(fits, indent=2), encoding='utf-8')
    writer.close()
    del base
    gc.collect()
    scores = pq.ParquetFile(output/'validation_scores.parquet').read().to_pandas()
    metric = PreparedEvaluation(scores, 2142).evaluate('score_online', threshold)
    report = {'selection': frozen, 'validation': metric, 'fits': fits,
              'scope': 'quarterly as-of refits, fixed2024threshold, open2025researchtransfer',
              'score_sha256': sha256(output/'validation_scores.parquet')}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:metric[k] for k in ('matched_episodes','emitted_warnings','episode_precision','full_episode_recall','full_episode_f1')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/ml-experiment'))
    parser.add_argument('--output', type=Path, default=Path('output/ml-experiment/online-linear'))
    args = parser.parse_args()
    run(args.root, args.output)
