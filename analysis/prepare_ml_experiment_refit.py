"""Append the sampled 2024 fold for a single post-selection final refit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from analysis.prepare_ml_experiment import sha256


def run(data: Path):
    output = data / 'refit_train.parquet'
    if output.exists():
        raise FileExistsError(output)
    def literal(path):
        return "'" + path.resolve().as_posix().replace("'", "''") + "'"
    with duckdb.connect(':memory:') as db:
        db.execute('SET threads=2')
        db.execute("SET memory_limit='1GB'")
        db.execute(f"""COPY (SELECT * FROM read_parquet({literal(data/'train.parquet')})
            UNION ALL SELECT * FROM read_parquet({literal(data/'tune.parquet')})
            WHERE target=1 OR hash(channel_id,prediction_time)%10000<200)
            TO {literal(output)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)""")
        stats = db.execute("""SELECT COUNT(*),SUM(target),COUNT(DISTINCT target_episode_id)
            FILTER(WHERE target=1),COUNT(*) FILTER(WHERE label_available_at >= TIMESTAMP '2025-01-01')
            FROM read_parquet(?)""", [str(output)]).fetchone()
    if stats[3]:
        raise ValueError('refit crosses the 2025 label boundary')
    manifest = {'rows': stats[0], 'positive_hours': stats[1], 'positive_episodes': stats[2],
                'file_sha256': sha256(output), 'train_source_sha256': sha256(data/'train.parquet'),
                'tune_source_sha256': sha256(data/'tune.parquet'),
                'fit_years': [2019, 2020, 2022, 2023, 2024],
                'sampling': 'all positives + deterministic2%negative hours, unchanged from primary experiment',
                'purpose': 'fit only after configuration and operating threshold are frozen on 2024'}
    (data/'refit_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('output/ml-experiment/data'))
    run(parser.parse_args().data)
