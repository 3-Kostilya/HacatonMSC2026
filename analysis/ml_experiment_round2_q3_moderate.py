"""Episode-equal, moderate-class-weight Q3 research model selected on 2024."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.ml_experiment_pooled import run


CONFIG = {"q3_episode_sqrt": {"engineered":True,"weights":"episode_sqrt",
                              "iterations":350,"depth":5}}


def episode_sqrt_weights(train: pd.DataFrame, strategy: str):
    if strategy != "episode_sqrt":
        raise ValueError(strategy)
    positive = train.target.eq(1)
    if train.target.isna().any() or not train.target.isin([0,1]).all():
        raise ValueError("sample has invalid target")
    if not positive.any() or positive.all():
        raise ValueError("both classes required")
    ids = train.loc[positive,"target_episode_id"]
    if ids.isna().any():
        raise ValueError("positive episode id required")
    counts = ids.value_counts()
    weight = np.ones(len(train),dtype="float32")
    weight[positive.to_numpy()] = 1/ids.map(counts).to_numpy(dtype="float32")
    weight[positive.to_numpy()] *= positive.sum()/weight[positive.to_numpy()].sum()
    ratio = (~positive).sum()/positive.sum()
    return weight,[1.0,float(np.sqrt(ratio))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("output/ml-experiment-round2/coverage/data"))
    parser.add_argument("--output",type=Path,default=Path("output/ml-experiment-round2/coverage/moderate"))
    args = parser.parse_args()
    run(args.data,args.output,configs_override=CONFIG,weight_builder=episode_sqrt_weights)
