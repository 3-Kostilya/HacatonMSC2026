"""Temporal linear ablations: stale counters, clipping and episode weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.ml_experiment_eval import PreparedEvaluation, search_thresholds


KEYS = ['channel_id', 'prediction_time', 'sensor_type', 'target',
        'target_episode_id', 'label_available_at']


def transform(frame: pd.DataFrame, names: list[str], variant: str, limits: dict) -> pd.DataFrame:
    out = frame[names].copy()
    out['sensor_type'] = out['sensor_type'].fillna('<unknown>')
    if variant == 'raw_balanced':
        return out
    for name in names:
        if name == 'sensor_type' or name.startswith('missing__'):
            continue
        values = pd.to_numeric(out[name], errors='coerce').astype('float32')
        if name in limits:
            values = values.clip(upper=limits[name])
        out[name] = np.log1p(values.clip(lower=0))
    for kind in ('event_count', 'alarm_count', 'state_transitions',
                 'registered_fault_text_count', 'normal_message_count'):
        for short, long in ((1, 6), (6, 24), (24, 168)):
            a = frame[f'{kind}_{short}h'].astype('float32')
            b = frame[f'{kind}_{long}h'].astype('float32')
            old_rate = (b - a).clip(lower=0) / (long - short)
            out[f'rate_change__{kind}_{short}_{long}'] = (
                np.log1p(a / short) - np.log1p(old_rate))
    return out


def score_file(source: Path, destination: Path, variants: list[dict], models: dict) -> None:
    writer = None
    for batch in pq.ParquetFile(source).iter_batches(batch_size=100_000):
        frame = batch.to_pandas()
        out = frame[KEYS].copy()
        for item in variants:
            name = item['name']
            out[f'score_{name}'] = models[name].predict_proba(
                transform(frame, item['names'], name, item['limits']))[:, 1].astype('float32')
        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(destination, table.schema, compression='zstd')
        writer.write_table(table)
    if writer is None:
        raise ValueError('empty score source')
    writer.close()


def run(data: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    train = pq.ParquetFile(data / 'train.parquet').read().to_pandas()
    base = manifest['all_features']
    variants = []
    models = {}
    for variant in ('raw_balanced', 'log_balanced', 'log_clip_deduplicated', 'log_episode_sqrt'):
        names = [x for x in base if not (variant in ('log_clip_deduplicated', 'log_episode_sqrt')
                                       and x == 'technical_message_count_168h')]
        limits = {}
        if variant in ('log_clip_deduplicated', 'log_episode_sqrt'):
            limits = {x: float(train[x].quantile(0.995)) for x in names
                      if 'message_count_' in x or 'fault_text_count_' in x}
        item = {'name': variant, 'names': names, 'limits': limits}
        x = transform(train, names, variant, limits)
        numeric = [n for n in x.columns if n != 'sensor_type']
        pre = ColumnTransformer([
            ('numeric', make_pipeline(SimpleImputer(strategy='constant', fill_value=-1),
                                     StandardScaler()), numeric),
            ('type', OneHotEncoder(handle_unknown='ignore'), ['sensor_type']),
        ], sparse_threshold=1.0)
        y = train.target.to_numpy(dtype='int8')
        weights = np.ones(len(train), dtype='float64')
        class_weight = 'balanced'
        if variant == 'log_episode_sqrt':
            positive = y == 1
            counts = train.loc[positive, 'target_episode_id'].value_counts()
            weights[positive] = 1 / train.loc[positive, 'target_episode_id'].map(counts).to_numpy()
            weights[positive] *= positive.sum() / weights[positive].sum()
            class_weight = {0: 1, 1: float(np.sqrt((y == 0).sum() / positive.sum()))}
        model = make_pipeline(pre, SGDClassifier(loss='log_loss', alpha=0.001,
                    class_weight=class_weight, random_state=42, max_iter=100, tol=1e-3))
        model.fit(x, y, sgdclassifier__sample_weight=weights)
        joblib.dump(model, output / f'{variant}.joblib')
        models[variant] = model
        variants.append(item)
        print('fitted', variant, 'seconds', round(time.perf_counter()-started, 1), flush=True)
        del x
    del train
    tune_path = output / 'tune_scores.parquet'
    score_file(data / 'tune.parquet', tune_path, variants, models)
    tune = pq.ParquetFile(tune_path).read().to_pandas()
    prepared = PreparedEvaluation(tune, manifest['full_episode_count']['tune'])
    selections = {}
    curves = {}
    for item in variants:
        name = item['name']
        curve = search_thresholds(prepared, f'score_{name}', points=41)
        curves[name] = curve
        selections[name] = max(curve, key=lambda r: (r['full_episode_f1'], r['full_episode_recall'], r['episode_precision']))
    winner = max(selections, key=lambda n: selections[n]['full_episode_f1'])
    # Write selection before opening transfer labels or scores.
    decision = {'selected_variant': winner, 'selections': selections, 'curves': curves,
                'variants': variants, 'selection_period': 2024, 'fit_end_exclusive': '2024-01-01'}
    (output / 'frozen_selection.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding='utf-8')
    del tune, prepared
    val_path = output / 'validation_scores.parquet'
    score_file(data / 'validation.parquet', val_path, variants, models)
    val = pq.ParquetFile(val_path).read().to_pandas()
    prepared = PreparedEvaluation(val, manifest['full_episode_count']['validation'])
    transfers = {n: prepared.evaluate(f'score_{n}', selections[n]['threshold']) for n in selections}
    report = {'status': 'research_open_2025_transfer', 'selected_variant': winner,
              'tune': selections, 'validation': transfers, 'elapsed_seconds': time.perf_counter()-started,
              'test_2026_read': False, 'source_data_manifest': str(data / 'manifest.json'),
              'selected_validation': transfers[winner]}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    compact = {n: {k: m[k] for k in ('matched_episodes', 'emitted_warnings',
                   'episode_precision', 'full_episode_recall', 'full_episode_f1')}
               for n, m in transfers.items()}
    print(json.dumps({'selected': winner, 'validation': compact}, ensure_ascii=False), flush=True)
    return report


def evaluate_saved(data: Path, output: Path) -> dict:
    """Mechanically replay 2024 selection with the canonical evaluator version."""
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    original = json.loads((output / 'frozen_selection.json').read_text(encoding='utf-8'))
    tune = pq.ParquetFile(output / 'tune_scores.parquet').read().to_pandas()
    prepared = PreparedEvaluation(tune, manifest['full_episode_count']['tune'])
    selections, curves = {}, {}
    for item in original['variants']:
        name = item['name']
        curve = search_thresholds(prepared, f'score_{name}', points=41)
        curves[name] = curve
        selections[name] = max(curve, key=lambda r: (r['full_episode_f1'], r['full_episode_recall'], r['episode_precision']))
    winner = max(selections, key=lambda n: selections[n]['full_episode_f1'])
    decision = {**original, 'selected_variant': winner, 'selections': selections, 'curves': curves,
                'reason': 'canonical float32 comparison replay; 2024-only selection'}
    (output / 'frozen_selection_canonical_v2.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding='utf-8')
    del tune, prepared
    val = pq.ParquetFile(output / 'validation_scores.parquet').read().to_pandas()
    prepared = PreparedEvaluation(val, manifest['full_episode_count']['validation'])
    transfers = {n: prepared.evaluate(f'score_{n}', selections[n]['threshold']) for n in selections}
    report = {'status': 'research_open_2025_transfer', 'selected_variant': winner,
              'tune': selections, 'validation': transfers,
              'test_2026_read': False, 'source_data_manifest': str(data / 'manifest.json'),
              'selected_validation': transfers[winner]}
    (output / 'report_canonical_v2.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    compact = {n: {k: m[k] for k in ('matched_episodes', 'emitted_warnings', 'episode_precision', 'full_episode_recall', 'full_episode_f1')} for n, m in transfers.items()}
    print(json.dumps({'selected': winner, 'validation': compact}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('output/ml-experiment/data'))
    parser.add_argument('--output', type=Path, default=Path('output/ml-experiment/linear'))
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    (evaluate_saved if args.evaluate_only else run)(args.data, args.output)
