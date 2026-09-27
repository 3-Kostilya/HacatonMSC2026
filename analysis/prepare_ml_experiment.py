"""Prepare versioned, purged temporal folds from accepted Q2 and B3 only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    package = Path('output/q2-a-full-sparse-20260926-v5')
    labels = Path('output/r3-b-full-months-20260925-v2')
    source = json.loads((package / 'manifest.json').read_text(encoding='utf-8'))
    if source['source_manifests']['b3'] != sha256(labels / 'manifest.json'):
        raise ValueError('B3 source hash differs')
    names = json.loads((package / 'model_feature_allowlist.json').read_text(encoding='utf-8'))['feature_names']
    output.mkdir(parents=True)
    writers = {}
    stats = {}
    full_episodes = {}
    started = time.perf_counter()
    with duckdb.connect(':memory:') as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        for record in source['months']:
            month = record['month']
            year = int(month[:4])
            if year not in (2019, 2020, 2022, 2023, 2024, 2025):
                raise ValueError(f'Forbidden month {month}')
            fold = 'validation' if year == 2025 else 'tune' if year == 2024 else 'train'
            boundary = '2026-01-01' if fold == 'validation' else '2025-01-01' if fold == 'tune' else '2024-01-01'
            rel = Path(f'year={year}/month={month[5:]}')
            fpath = package / rel / 'model_features.parquet'
            lpath = labels / rel / 'registered_forecast_labels.parquet'
            truth = db.execute("""SELECT DISTINCT target_episode_id, sensor_type
                FROM read_parquet(?) WHERE target=1 AND split_status='assigned'
                AND label_available_at < CAST(? AS TIMESTAMP)""", [str(lpath), boundary]).fetchall()
            full_episodes.setdefault(fold, {}).update(dict(truth))
            sample = "AND (l.target=1 OR hash(f.channel_id,f.prediction_time)%10000 < 200)" if fold == 'train' else ''
            select = ','.join('f."' + x + '"' for x in names)
            query = f"""SELECT f.channel_id,f.prediction_time,{select},
                l.target,l.target_episode_id,l.label_available_at
                FROM read_parquet(?) f JOIN read_parquet(?) l USING(channel_id,prediction_time)
                WHERE l.split_status='assigned' AND l.target IN (0,1)
                AND l.label_available_at < CAST(? AS TIMESTAMP)
                AND f.sensor_type IS NOT DISTINCT FROM l.sensor_type {sample}"""
            reader = db.execute(query, [str(fpath), str(lpath), boundary]).to_arrow_reader(batch_size=100_000)
            for batch in reader:
                table = pa.Table.from_batches([batch])
                if fold not in writers:
                    writers[fold] = pq.ParquetWriter(output / f'{fold}.parquet', table.schema, compression='zstd')
                    stats[fold] = {'rows': 0, 'positive_hours': 0}
                writers[fold].write_table(table)
                stats[fold]['rows'] += table.num_rows
                stats[fold]['positive_hours'] += sum(table['target'].to_pylist())
            print(month, fold, stats.get(fold), flush=True)
        for writer in writers.values():
            writer.close()
    # Refit data are deliberately omitted: frozen models selected on 2024 retain
    # their score scale for 2025. Additional future runs can add explicit refits.
    manifest = {
        'schema_version': 'ml-experiment-temporal-folds-v1',
        'all_features': names,
        'full_episode_count': {k: len(v) for k, v in full_episodes.items()},
        'full_episode_count_by_type': {k: {t: list(v.values()).count(t) for t in set(v.values())} for k, v in full_episodes.items()},
        'row_stats': stats,
        'train_years': [2019, 2020, 2022, 2023],
        'tune_year': 2024,
        'validation_year': 2025,
        'negative_sampling_fraction': 0.02,
        'boundary_rule': 'label_available_at strictly before next fold boundary',
        'source_hashes': {'q2': sha256(package / 'manifest.json'), 'b3': sha256(labels / 'manifest.json')},
        'elapsed_seconds': time.perf_counter() - started,
        'test_2026_read': False,
        'validation_2025_is_already_open': True,
        'files': {k: {'path': f'{k}.parquet', 'sha256': sha256(output / f'{k}.parquet')} for k in writers},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('all_features', 'full_episode_count_by_type')}, ensure_ascii=False), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/ml-experiment/data'))
    run(parser.parse_args().output)
