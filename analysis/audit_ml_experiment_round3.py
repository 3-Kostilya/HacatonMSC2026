"""Independent saved-warning audit of research recovery-reset experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from analysis.prepare_ml_experiment import sha256


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(root: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    checks = []
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        for directory in sorted(root.glob("recovery-q2-*")):
            report_path = directory / "report.json"
            if not report_path.exists():
                continue
            report = read(report_path)
            source = Path(report["source"])
            if sha256(source) != report["source_sha256"]:
                raise AssertionError(f"score file changed: {directory}")
            for family in ["control", "candidate"]:
                warnings = directory / f"{family}_warnings.parquet"
                rows, matched, unique_ids, duplicates, resets, bad = db.execute(f'''
                    SELECT COUNT(*),COUNT(*) FILTER(WHERE w.outcome='matched_episode'),
                    COUNT(DISTINCT w.target_episode_id) FILTER(WHERE w.outcome='matched_episode'),
                    COUNT(*)-COUNT(DISTINCT (w.channel_id,w.prediction_time)),
                    COUNT(*) FILTER(WHERE w.reason='recovered_episode_reset'),
                    COUNT(*) FILTER(WHERE s.channel_id IS NULL
                        OR w.sensor_type IS DISTINCT FROM s.sensor_type
                        OR w.target IS DISTINCT FROM s.target
                        OR w.target_episode_id IS DISTINCT FROM s.target_episode_id
                        OR w.label_available_at IS DISTINCT FROM s.label_available_at
                        OR s."{report['column']}" < ?)
                    FROM read_parquet(?) w LEFT JOIN read_parquet(?) s
                    USING(channel_id,prediction_time)''',
                    [report["threshold"], str(warnings), str(source)]).fetchone()
                expected = report[family]
                if (rows != expected["emitted_warnings"] or matched != expected["matched_episodes"]
                        or unique_ids != matched or duplicates or bad
                        or (family == "candidate" and resets != report["reset_warnings"])
                        or (family == "control" and resets)):
                    raise AssertionError(f"saved warning/label parity differs: {directory}/{family}")
                reset_bad = db.execute('''SELECT COUNT(*) FROM read_parquet(?)
                    WHERE reason='recovered_episode_reset' AND NOT (
                    previous_warning_at < observed_onset_at
                    AND observed_onset_at < observed_recovery_at
                    AND observed_recovery_at <= prediction_time
                    AND observed_onset_at <= previous_warning_at + INTERVAL '24 hours')''',
                    [str(warnings)]).fetchone()[0]
                if reset_bad:
                    raise AssertionError(f"future reset witness: {directory}/{family}")
                checks.append({"experiment": directory.name, "family": family,
                               "warnings": rows, "matched_episodes": matched,
                               "reset_warnings": resets, "metadata_mismatches": bad,
                               "duplicate_warning_keys": duplicates,
                               "future_reset_witnesses": reset_bad})
    result = {"status": "independent_recovery_warning_parity_passed", "checks": checks,
              "test_2026_read": False, "data_2021_read": False}
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"independent recovery warning checks passed: {len(checks)}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/ml-experiment-round3"))
    parser.add_argument("--output", type=Path, default=Path("output/ml-experiment-round3/independent-audit"))
    args = parser.parse_args()
    run(args.root, args.output)
