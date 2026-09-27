"""Causal longer-term recurrence features from already completed B2 episodes.

The catalog is retrospective, but only exact, unambiguous closed episodes enter
features, and an episode is invisible until end_at. No outcome counts from the
prediction horizon enter a snapshot. History never crosses excluded 2021.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class CompletedHistory:
    def __init__(self, catalog: pd.DataFrame):
        valid = catalog.loc[
            catalog.onset_status.eq('candidate_new_onset')
            & catalog.end_status.eq('exact_norma')
            & ~catalog.uncertain_intervening_state.fillna(True)
            & catalog.end_at.notna()
            & catalog.start_at.notna()
        ].copy()
        valid = valid.loc[valid.start_at.dt.year.ne(2021) & valid.end_at.dt.year.ne(2021)
                          & valid.end_at.lt(pd.Timestamp('2026-01-01'))
                          & valid.end_at.ge(valid.start_at)]
        # Exclude any closure that straddles the missing archive segment.
        valid = valid.loc[(valid.start_at.lt('2021-01-01') & valid.end_at.lt('2021-01-01'))
                          | (valid.start_at.ge('2022-01-01') & valid.end_at.ge('2022-01-01'))]
        self.by_channel = {}
        for channel, group in valid.groupby('channel_id', sort=False):
            group = group.sort_values('end_at')
            ends = group.end_at.to_numpy(dtype='datetime64[ns]').astype('int64')
            durations = ((group.end_at - group.start_at).dt.total_seconds().to_numpy())
            self.by_channel[str(channel)] = (ends, np.r_[0.0, np.cumsum(durations)])

    @classmethod
    def load(cls, root: Path = Path('output/r2-b-registered-episodes-full-r1v2')):
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        path = root / 'registered_state_episodes.parquet'
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != manifest['files'][path.name]['sha256']:
            raise ValueError('B2 catalog hash differs')
        columns = ['channel_id', 'start_at', 'end_at', 'onset_status', 'end_status',
                   'uncertain_intervening_state']
        # Filter values at source; 2026 outcomes are never included in history.
        filters = [
            [('start_at', '<', pd.Timestamp('2021-01-01')),
             ('end_at', '<', pd.Timestamp('2021-01-01'))],
            [('start_at', '>=', pd.Timestamp('2022-01-01')),
             ('end_at', '>=', pd.Timestamp('2022-01-01')),
             ('end_at', '<', pd.Timestamp('2026-01-01'))],
        ]
        frame = pq.read_table(path, columns=columns, filters=filters).to_pandas()
        return cls(frame)

    def features(self, frame: pd.DataFrame) -> pd.DataFrame:
        n = len(frame)
        result = {f'history_completed_count_{days}d': np.zeros(n, dtype='float32')
                  for days in (7, 28, 90, 365)}
        result.update({f'history_mean_duration_{days}d': np.full(n, np.nan, dtype='float32')
                       for days in (28, 90)})
        result.update({
            'history_completed_count_segment': np.zeros(n, dtype='float32'),
            'history_last_end_age_seconds': np.full(n, np.nan, dtype='float32'),
            'history_previous_end_interval_seconds': np.full(n, np.nan, dtype='float32'),
        })
        times = frame.prediction_time.to_numpy(dtype='datetime64[ns]').astype('int64')
        boundary = np.where(times < pd.Timestamp('2021-01-01').value,
                            pd.Timestamp('2019-01-01').value, pd.Timestamp('2022-01-01').value)
        for channel, positions in frame.groupby('channel_id', sort=False).indices.items():
            if str(channel) not in self.by_channel:
                continue
            ends, duration_sum = self.by_channel[str(channel)]
            at = times[positions]
            segment = boundary[positions]
            upper = np.searchsorted(ends, at, side='right')
            lower_segment = np.searchsorted(ends, segment, side='left')
            counts = upper - lower_segment
            result['history_completed_count_segment'][positions] = counts
            for days in (7, 28, 90, 365):
                lower = np.searchsorted(ends, np.maximum(at-days*86400*10**9, segment), side='right')
                count = upper-lower
                result[f'history_completed_count_{days}d'][positions] = count
                if days in (28, 90):
                    total_duration = duration_sum[upper]-duration_sum[lower]
                    mean = np.divide(total_duration, count, out=np.full(len(count), np.nan), where=count>0)
                    result[f'history_mean_duration_{days}d'][positions] = mean
            have = counts > 0
            result['history_last_end_age_seconds'][positions[have]] = (at[have]-ends[upper[have]-1])/10**9
            two = counts > 1
            result['history_previous_end_interval_seconds'][positions[two]] = (
                ends[upper[two]-1]-ends[upper[two]-2])/10**9
        out = pd.DataFrame(result, index=frame.index)
        for name in list(out):
            out[f'log__{name}'] = np.log1p(out[name].clip(lower=0)).astype('float32')
        return out
