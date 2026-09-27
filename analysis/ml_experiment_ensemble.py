"""Compare frozen cross-family agreements using only 2024 to choose a policy."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.ml_experiment_eval import PreparedEvaluation, search_thresholds
from analysis.prepare_ml_experiment import sha256


KEYS = ['channel_id', 'prediction_time', 'sensor_type', 'target',
        'target_episode_id', 'label_available_at']


def logit_margin(scores: np.ndarray, threshold: float) -> np.ndarray:
    p = np.clip(np.asarray(scores, dtype='float64'), 1e-7, 1-1e-7)
    q = np.clip(threshold, 1e-7, 1-1e-7)
    return (np.log(p/(1-p))-np.log(q/(1-q))).astype('float32')


def read_family_spec(root: Path) -> list[dict]:
    families = []
    for kind in ('pooled', 'history'):
        folder = root / kind
        report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
        name = report['selected_variant']
        chosen = report['experiments'][name]['tune']['selected']
        families.append({'name': kind, 'folder': str(folder), 'column': f'score_{name}',
                         'threshold': chosen['threshold'],
                         'tune_file': 'scores_tune.parquet', 'validation_file': 'scores_validation.parquet'})
    folder = root / 'linear'
    report = json.loads((folder / 'report_canonical_v2.json').read_text(encoding='utf-8'))
    name = report['selected_variant']
    families.append({'name': 'linear', 'folder': str(folder), 'column': f'score_{name}',
                     'threshold': report['tune'][name]['threshold'],
                     'tune_file': 'tune_scores.parquet', 'validation_file': 'validation_scores.parquet'})
    return families


def build_scores(families: list[dict], fold: str, destination: Path) -> None:
    paths = [str(Path(x['folder']) / x[f'{fold}_file']) for x in families]
    select = ','.join(f'a.{key}' for key in KEYS)
    select += ',' + ','.join(f'{chr(97+i)}."{item["column"]}" AS raw_{i}' for i, item in enumerate(families))
    sources = 'read_parquet(?) a ' + ' '.join(
        f'JOIN read_parquet(?) {chr(97+i)} USING(channel_id,prediction_time)' for i in range(1, len(families)))
    writer = None
    rows = 0
    with duckdb.connect(':memory:') as db:
        db.execute('SET threads=2')
        db.execute("SET memory_limit='2GB'")
        reader = db.execute(f'SELECT {select} FROM {sources}', paths).to_arrow_reader(batch_size=100_000)
        for batch in reader:
            frame = batch.to_pandas()
            out = frame[KEYS].copy()
            margins = np.column_stack([logit_margin(frame[f'raw_{i}'].to_numpy(), item['threshold'])
                                       for i, item in enumerate(families)])
            # Fixed equally weighted agreement rules, not a label-fitted meta model.
            out['score_mean'] = margins.mean(axis=1).astype('float32')
            out['score_min'] = margins.min(axis=1).astype('float32')
            out['score_median'] = np.median(margins, axis=1).astype('float32')
            out['score_trees_mean'] = margins[:, :2].mean(axis=1).astype('float32')
            out['score_trees_min'] = margins[:, :2].min(axis=1).astype('float32')
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression='zstd')
            writer.write_table(table)
            rows += len(frame)
        expected = db.execute('SELECT COUNT(*) FROM read_parquet(?)', [paths[0]]).fetchone()[0]
    if writer is None or rows != expected:
        raise ValueError('Ensemble source keys differ or source is empty')
    writer.close()


def run(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    families = read_family_spec(root)
    output.mkdir(parents=True)
    manifest = json.loads((root / 'data/manifest.json').read_text(encoding='utf-8'))
    build_scores(families, 'tune', output / 'tune_scores.parquet')
    tune = pq.ParquetFile(output / 'tune_scores.parquet').read().to_pandas()
    prepared = PreparedEvaluation(tune, manifest['full_episode_count']['tune'])
    curves = {}
    selections = {}
    for column in ('score_mean', 'score_min', 'score_median', 'score_trees_mean', 'score_trees_min'):
        curves[column] = search_thresholds(prepared, column, points=45)
        selections[column] = max(curves[column], key=lambda m: (m['full_episode_f1'], m['episode_precision']))
    winner = max(selections, key=lambda c: selections[c]['full_episode_f1'])
    decision = {'selected_column': winner, 'frozen_threshold': selections[winner]['threshold'],
                'families': families, 'selection_year': 2024, 'tune': selections, 'curves': curves}
    (output / 'selection.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding='utf-8')
    del prepared, tune
    gc.collect()
    build_scores(families, 'validation', output / 'validation_scores.parquet')
    val = pq.ParquetFile(output / 'validation_scores.parquet').read().to_pandas()
    prepared = PreparedEvaluation(val, manifest['full_episode_count']['validation'])
    transfers = {column: prepared.evaluate(column, chosen['threshold']) for column, chosen in selections.items()}
    report = {'selected_column': winner, 'selected_validation': transfers[winner],
              'tune': selections, 'validation': transfers, 'families': families,
              'scope': '2024 selection, frozen open 2025 temporal transfer',
              'artifact_hashes': {name: sha256(output/name)
                                  for name in ('selection.json', 'tune_scores.parquet', 'validation_scores.parquet')}}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({c: {k: m[k] for k in ('episode_precision','full_episode_recall','full_episode_f1')}
                      for c,m in transfers.items()}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/ml-experiment'))
    parser.add_argument('--output', type=Path, default=Path('output/ml-experiment/ensemble'))
    args = parser.parse_args()
    run(args.root, args.output)
