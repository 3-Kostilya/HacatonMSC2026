"""Compare Q1/R4 and Q2 scores on exactly the same validation keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.metrics import average_precision_score

from analysis.train_r4_discrete_baselines import read_json, sha256


def audit(*, q1_dir: Path, q2_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    q1 = read_json(q1_dir / "manifest.json")
    q2 = read_json(q2_dir / "manifest.json")
    if (q1["schema_version"] != "q1-b-fixed-population-qa-ablation-v1"
            or q2["schema_version"] != "q2-b-expanded-validation-v1"):
        raise ValueError("Q1/Q2 experiment versions differ")
    parts = []
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=2")
        for index in range(1, 13):
            label = f"2025-{index:02d}"
            old = q1_dir / f"validation_{label}.parquet"
            new = q2_dir / f"validation_{label}.parquet"
            old_count = db.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(old)]).fetchone()[0]
            frame = db.execute("""SELECT q1.target, q1.target_episode_id,
                q1.score_baseline AS old_r4, q2.score_base51 AS q2_base51,
                q2.score_full121 AS q2_full121, q2.score_linear121 AS q2_linear121,
                q2.target AS new_target, q2.target_episode_id AS new_episode
                FROM read_parquet(?) q1 JOIN read_parquet(?) q2
                USING(channel_id,prediction_time)""", [str(old), str(new)]).fetch_df()
            if (not frame.target.eq(frame.new_target).all()
                    or not frame.target_episode_id.fillna("<null>").eq(
                        frame.new_episode.fillna("<null>")).all()
                    or len(frame) > old_count):
                raise ValueError(f"Q1/Q2 label or key differs: {label}")
            parts.append(frame)
    common = pd.concat(parts, ignore_index=True)
    names = ("old_r4", "q2_base51", "q2_full121", "q2_linear121")
    result = {
        "schema_version": "q2-b-common-population-audit-v1",
        "q1_manifest_sha256": sha256(q1_dir / "manifest.json"),
        "q2_manifest_sha256": sha256(q2_dir / "manifest.json"),
        "old_validation_rows": 1_365_077,
        "common_rows": len(common),
        "common_positive_hours": int(common.target.sum()),
        "common_positive_episodes": int(common.loc[common.target == 1,
                                                "target_episode_id"].nunique()),
        "hourly_average_precision": {
            name: float(average_precision_score(common.target, common[name]))
            for name in names
        },
        "interpretation": "same validation keys and unchanged B3 labels; ranking comparison only",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("q1-dir", "q2-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = audit(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
