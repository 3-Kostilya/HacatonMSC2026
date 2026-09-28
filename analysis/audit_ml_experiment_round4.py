"""Independent full-context warning and unknown-outcome parity audit."""
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
    q3 = Path("output/q3-a-coverage-reentry-20260927-v4")
    published = read(q3/"report.json")
    label_files=[]
    for month in range(1,13):
        path=Path(f"output/r3-b-full-months-20260925-v2/year=2025/month={month:02d}/registered_forecast_labels.parquet")
        expected=next(item["sha256"] for item in published["source_b3_labels"]
                      if item["month"]==f"2025-{month:02d}")
        if sha256(path)!=expected:
            raise AssertionError("published B3 label file changed")
        label_files.append(str(path))
    checks=[]
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        for directory in sorted(root.glob("q3-full-context-*")):
            report_path=directory/"report.json"
            if not report_path.exists():
                continue
            report=read(report_path)
            source=directory/"full_context_scores.parquet"
            if sha256(source)!=report["full_context_scores_sha256"]:
                raise AssertionError("full score context changed")
            for family in ["control","candidate"]:
                warning=directory/f"{family}_warnings.parquet"
                expected=report[family]
                rows,matched,unique,unknown,negative,duplicate,keys_bad,metadata_bad=db.execute('''
                    SELECT COUNT(*),
                    COUNT(*) FILTER(WHERE w.outcome='matched_known_episode'),
                    COUNT(DISTINCT w.target_episode_id)
                        FILTER(WHERE w.outcome='matched_known_episode'),
                    COUNT(*) FILTER(WHERE w.outcome='unknown_target'),
                    COUNT(*) FILTER(WHERE w.outcome='known_no_target'),
                    COUNT(*)-COUNT(DISTINCT(w.channel_id,w.prediction_time)),
                    COUNT(*) FILTER(WHERE s.channel_id IS NULL OR s.sensor_type IS DISTINCT FROM w.sensor_type
                        OR s.score < ?),
                    COUNT(*) FILTER(WHERE b.channel_id IS NULL OR b.sensor_type IS DISTINCT FROM w.sensor_type
                        OR b.target IS DISTINCT FROM w.target
                        OR b.target_episode_id IS DISTINCT FROM w.target_episode_id
                        OR b.label_available_at IS DISTINCT FROM w.label_available_at)
                    FROM read_parquet(?) w LEFT JOIN read_parquet(?) s
                    USING(channel_id,prediction_time)
                    LEFT JOIN read_parquet(?,hive_partitioning=false) b
                    USING(channel_id,prediction_time)''',
                    [report["threshold"],str(warning),str(source),label_files]).fetchone()
                if (rows!=expected["warnings"] or matched!=expected["matched_known_episodes"]
                        or matched!=unique or unknown!=expected["unknown_outcome_warnings"]
                        or negative!=expected["known_no_target_warnings"]
                        or expected["precision_lower_bound"]!=matched/rows
                        or expected["full_recall_lower_bound"]!=matched/2142
                        or expected["precision_loose_upper_bound"]!=min(2142,matched+unknown)/rows
                        or duplicate or keys_bad or metadata_bad):
                    raise AssertionError(f"independent full-context warning parity differs: {directory}/{family}")
                reset_bad=0
                if family=="candidate":
                    reset_bad=db.execute('''SELECT COUNT(*) FROM read_parquet(?) WHERE
                        reason='recovered_episode_reset' AND NOT(
                        previous_warning_at<observed_onset_at
                        AND observed_onset_at<observed_recovery_at
                        AND observed_recovery_at<=prediction_time
                        AND observed_onset_at<=previous_warning_at+INTERVAL '24 hours')''',
                        [str(warning)]).fetchone()[0]
                if reset_bad:
                    raise AssertionError("future recovery witness")
                checks.append({"experiment":directory.name,"family":family,
                               "warnings":rows,"matched_known_episodes":matched,
                               "unknown_outcome_warnings":unknown,"metadata_mismatches":metadata_bad,
                               "future_recovery_witnesses":reset_bad})
    result={"status":"independent_full_context_warning_and_unknown_parity_passed",
            "checks":checks,"test_2026_read":False,"data_2021_read":False}
    (output/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"independent full-context warning checks passed: {len(checks)}",flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("output/ml-experiment-round4"))
    parser.add_argument("--output",type=Path,default=Path("output/ml-experiment-round4/independent-audit"))
    args=parser.parse_args()
    run(args.root,args.output)
