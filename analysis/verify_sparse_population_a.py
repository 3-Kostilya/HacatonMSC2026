"""Independent full-export, missingness, labels and M1 stream checks for Q2 A."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from itertools import groupby
from pathlib import Path
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.build_a2_hourly import _monthly_files
from analysis.build_quality_improvement_a import QA_NAMES, YEARS, expected_months, quoted, safe_path
from analysis.build_sparse_population_a import write_json
from analysis.replay_shadow_pilot import COLUMNS
from analysis.r6_provenance import frozen_rule_sha256
from analysis.train_r4_discrete_baselines import read_json, sha256
from stage1.features.sparse_admission import SparseAdmissionStream
from stage1.features.qa_values import qa_window_counts


def active_package(package):
    return package if package.exists() else package.with_name(package.name + ".inprogress")


def await_json(package, relative):
    """Wait for a completed builder file; the final root is renamed atomically."""
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        root = active_package(package)
        path = root / relative
        try:
            return root, read_json(path)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(2)
    raise TimeoutError(f"builder has not published {relative} within 15 minutes")


def month_snapshots(package, follow_build):
    if not follow_build:
        items = read_json(package / "manifest.json")["months"]
        if [item["month"] for item in items] != expected_months():
            raise ValueError("full package months differ before any label/feature reads")
        for item in items:
            yield package, item
        return
    months = expected_months()
    for label in months:
        relative = f"year={label[:4]}/month={label[5:]}/manifest.json"
        if label == months[-1]:
            # Do not keep Parquet files open across the builder's final rename.
            await_json(package, "manifest.json")
        root, item = await_json(package, relative)
        yield root, {**item, "manifest_file": relative, "manifest_sha256": sha256(root / relative)}


def verify(*, package, m1_dir, a3_dir, b3_dir, corrections_dir, output, follow_build=False):
    if output.exists():
        raise FileExistsError(output)
    q1 = read_json(Path("ml/quality_improvement_feature_contract_v1.json"))
    old_allowlist = read_json(Path("ml/r3_discrete_feature_allowlist_v1.json"))
    pins = {
        "m1": q1["source_m1_manifest_sha256"],
        "a3": q1["source_a3_manifest_sha256"],
        "b3": old_allowlist["source_b3_manifest_sha256"],
        "corrections": q1["source_qa_correction_features_sha256"],
    }
    if sha256(m1_dir / "manifest.json") != pins["m1"]:
        raise ValueError("M1 differs")
    if sha256(b3_dir / "manifest.json") != pins["b3"]:
        raise ValueError("B3 differs")
    if sha256(a3_dir / "manifest.json") != pins["a3"]:
        raise ValueError("A3 differs")
    corrections = corrections_dir / "feature_corrections.parquet"
    if sha256(corrections) != pins["corrections"]:
        raise ValueError("QA corrections differ")
    if follow_build:
        _, allowlist = await_json(package, "model_feature_allowlist.json")
    else:
        allowlist = read_json(package / "model_feature_allowlist.json")
    base = allowlist["base_feature_names"]
    expected_masks = [f"missing__{name}" for name in old_allowlist["feature_names"]
                      if name != "sensor_type"]
    if (base != old_allowlist["feature_names"]
            or allowlist["feature_names"] != [*base, *QA_NAMES, *expected_masks]):
        raise ValueError("full allowlist differs from agreed source features and masks")
    mask_sql = " OR ".join(
        f'"missing__{n}" IS DISTINCT FROM CAST("{n}" IS NULL AS TINYINT)'
        for n in base
        if n != "sensor_type"
    )
    month_results = []
    verified_snapshots = []
    samples = []
    bchunks = {c["month"]: c for c in read_json(b3_dir / "manifest.json")["chunks"]}
    achunks = {c["month"]: c for c in read_json(a3_dir / "manifest.json")["chunks"]}
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        db.execute(
            "CREATE TEMP VIEW corrections AS SELECT * FROM read_parquet("
            + quoted(str(corrections)) + ")"
        )
        for root, month in month_snapshots(package, follow_build):
            label = month["month"]
            folder = root / f"year={label[:4]}" / f"month={label[5:]}"
            if sha256(folder / "manifest.json") != month["manifest_sha256"]:
                raise ValueError("full monthly manifest differs")
            for name, info in month["files"].items():
                if sha256(folder / name) != info["sha256"]:
                    raise ValueError("full monthly Parquet hash differs")
            admission = folder / "admission.parquet"
            features = folder / "model_features.parquet"
            labels = (
                b3_dir
                / Path(bchunks[label]["manifest_file"]).parent
                / "registered_forecast_labels.parquet"
            )
            bm = read_json(labels.parent / "manifest.json")
            if sha256(labels.parent / "manifest.json") != bchunks[label]["manifest_sha256"]:
                raise ValueError("B3 monthly manifest differs")
            if sha256(labels) != bm["files"][labels.name]["sha256"]:
                raise ValueError("B3 label file hash differs")
            db.execute(
                "CREATE OR REPLACE TEMP VIEW d AS SELECT * FROM read_parquet("
                + quoted(str(admission))
                + ")"
            )
            db.execute(
                "CREATE OR REPLACE TEMP VIEW f AS SELECT * FROM read_parquet("
                + quoted(str(features))
                + ")"
            )
            db.execute(
                "CREATE OR REPLACE TEMP VIEW l AS SELECT * FROM read_parquet("
                + quoted(str(labels))
                + ")"
            )
            count, unique, bad = db.execute(
                "SELECT COUNT(*),COUNT(DISTINCT (channel_id,prediction_time)),"
                "COUNT(*) FILTER(WHERE (admission_status='eligible' AND (len(admission_reasons)>0 OR "
                "blocking_qa_count_24h>0 OR last_explicit_normal_at IS NULL OR "
                "last_explicit_normal_at<prediction_time-INTERVAL '168 hours')) OR "
                "admission_evidence_through>prediction_time) FROM d"
            ).fetchone()
            labeled_count = db.execute(
                "SELECT COUNT(*) FROM d JOIN l USING(channel_id,prediction_time)"
            ).fetchone()[0]
            missing = db.execute(
                "SELECT COUNT(*) FROM d ANTI JOIN l USING(channel_id,prediction_time)"
            ).fetchone()[0]
            mask_bad = db.execute("SELECT COUNT(*) FROM f WHERE " + mask_sql).fetchone()[0]
            source_features = safe_path(a3_dir, achunks[label]["features_file"], month=label)
            if sha256(source_features) != achunks[label]["features_sha256"]:
                raise ValueError("A3 feature content differs")
            differences = " OR ".join(
                f'f."{name}" IS DISTINCT FROM CASE WHEN c.channel_id IS NOT NULL '
                f'THEN c."{name}" ELSE a."{name}" END'
                for name in base
            )
            base_bad = db.execute(
                "SELECT COUNT(*) FROM f LEFT JOIN read_parquet(?) a "
                "USING(channel_id,prediction_time) LEFT JOIN corrections c "
                "USING(channel_id,prediction_time) WHERE a.channel_id IS NULL OR "
                + differences, [str(source_features)]
            ).fetchone()[0]
            qa_bad = db.execute(
                "SELECT COUNT(*) FROM f WHERE "
                + " OR ".join(f'"{name}" IS NULL OR "{name}"<0' for name in QA_NAMES)
            ).fetchone()[0]
            feature_count, feature_unique = db.execute(
                "SELECT COUNT(*),COUNT(DISTINCT (channel_id,prediction_time)) FROM f"
            ).fetchone()
            eligible = db.execute(
                "SELECT COUNT(*) FROM d WHERE admission_status='eligible'"
            ).fetchone()[0]
            unexpected = db.execute(
                "SELECT COUNT(*) FROM f LEFT JOIN d USING(channel_id,prediction_time) "
                "WHERE d.channel_id IS NULL OR d.admission_status<>'eligible' "
                "OR f.sensor_type IS DISTINCT FROM d.sensor_type"
            ).fetchone()[0]
            if (
                count != month["decision_rows"]
                or unique != count
                or bad
                or missing
                or labeled_count != count
                or feature_count != eligible
                or feature_unique != eligible
                or mask_bad
                or base_bad
                or qa_bad
                or unexpected
            ):
                raise ValueError(
                    f"full export, label keys, missingness or guard check failed: {label}"
                )
            if pq.ParquetFile(features).schema_arrow.names != [
                "channel_id",
                "prediction_time",
                *allowlist["feature_names"],
            ]:
                raise ValueError("feature schema includes unapproved fields")
            # Past-only decisions select verification examples, never training keys.
            selected = (
                db.execute(
                    "SELECT channel_id,prediction_time,sensor_type,admission_status,admission_reasons,"
                    "last_explicit_normal_at,blocking_qa_count_24h FROM d QUALIFY "
                    "ROW_NUMBER() OVER(PARTITION BY admission_status ORDER BY sha256(channel_id),prediction_time)=1"
                )
                .to_arrow_table()
                .to_pylist()
            )
            samples.extend(selected)
            # Include rare nonzero QA examples when present, selected without labels.
            qa_case = db.execute(
                "SELECT d.channel_id,d.prediction_time,d.sensor_type,d.admission_status,"
                "d.admission_reasons,d.last_explicit_normal_at,d.blocking_qa_count_24h "
                "FROM d JOIN f USING(channel_id,prediction_time) WHERE "
                + " OR ".join(f'f."{name}">0' for name in QA_NAMES)
                + " ORDER BY sha256(d.channel_id),d.prediction_time LIMIT 1"
            ).to_arrow_table().to_pylist()
            samples.extend(qa_case)
            month_results.append(
                {
                    "month": label,
                    "decision_rows": count,
                    "feature_rows": feature_count,
                    "label_key_mismatches": 0,
                    "missingness_mismatches": 0,
                    "base_feature_mismatches": 0,
                    "qa_count_violations": 0,
                    "guard_violations": 0,
                }
            )
            verified_snapshots.append((label, month["manifest_sha256"]))
            print(f"verified {label}: {count} decisions / {feature_count} feature rows", flush=True)
        if [r["month"] for r in month_results] != expected_months():
            raise ValueError("full package does not contain exactly 72 accepted months")
        # Final manifest ties every already checked month to the published result.
        manifest = read_json(package / "manifest.json")
        report = read_json(package / "report.json")
        if manifest["source_manifests"] != pins or verified_snapshots != [
            (m["month"], m["manifest_sha256"]) for m in manifest["months"]
        ]:
            raise ValueError("published source pins/months differ from verified snapshots")
        for name, info in manifest["files"].items():
            if sha256(safe_path(package, name)) != info["sha256"]:
                raise ValueError("full package file hash differs")
        if any(frozen_rule_sha256(Path(name)) != pin for name, pin in report["code_lf_sha256"].items()):
            raise ValueError("calculation code differs from pinned source")
        for source in report["source_m1_files"]:
            if sha256(safe_path(m1_dir, source["file"])) != source["sha256"]:
                raise ValueError("M1 content differs from calculation")
        positives = pq.read_table(package / "positive_hour_diagnostics.parquet").to_pylist()
        identities = [(r["channel_id"], r["prediction_time"]) for r in positives]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate positive diagnostic key")
        counts = Counter(r["split"] for r in positives)
        if counts != {"train": 75629, "validation": 19588}:
            raise ValueError("positive hour totals differ from accepted B3")
        episode_counts = Counter(
            r["split"] for r in pq.read_table(package / "episode_diagnostics.parquet").to_pylist()
        )
        if episode_counts != {"train": 7138, "validation": 2142}:
            raise ValueError("episode totals differ from accepted B3")
        # Hash-ranked channel subset bounds raw replay costs. It includes the
        # unknown/excluded statuses; no future labels select the audit sample.
        channels = sorted(
            {s["channel_id"] for s in samples},
            key=lambda c: hashlib.sha256(c.encode()).hexdigest(),
        )[:20]
        samples = [s for s in samples if s["channel_id"] in channels]
        samples = list({(s["channel_id"], s["prediction_time"]): s for s in samples}.values())
        sampled_features = {}
        for month in manifest["months"]:
            label = month["month"]
            month_keys = [
                {"channel_id": s["channel_id"], "prediction_time": s["prediction_time"]}
                for s in samples
                if s["prediction_time"].strftime("%Y-%m") == label
                and s["admission_status"] == "eligible"
            ]
            if not month_keys:
                continue
            db.register("sample_keys", pa.Table.from_pylist(month_keys))
            feature_file = package / f"year={label[:4]}" / f"month={label[5:]}" / "model_features.parquet"
            for row in db.execute(
                "SELECT f.* FROM read_parquet(?) f SEMI JOIN sample_keys "
                "USING(channel_id,prediction_time)", [str(feature_file)]
            ).to_arrow_table().to_pylist():
                sampled_features[row["channel_id"], row["prediction_time"]] = row
        files = []
        last = max(s["prediction_time"] for s in samples)
        for year in YEARS:
            if year > last.year:
                continue
            fs, absent = _monthly_files(m1_dir, datetime(year, 1, 1), datetime(year + 1, 1, 1))
            if absent:
                raise ValueError("raw replay month missing")
            files.extend(fs)
        db.register("selected_channels", pa.table({"channel_id": channels}))
        reader = db.execute(
            "SELECT " + ",".join(COLUMNS) + " FROM read_parquet(?,hive_partitioning=false) "
            "SEMI JOIN selected_channels USING(channel_id) WHERE timestamp<=? AND "
            "split_part(replace(source,chr(92),'/'),'/',-1)="
            "'ext-journal-' || CAST(year(timestamp) AS VARCHAR) || '.7z' ORDER BY timestamp,channel_id,row_id",
            [[str(p) for p in files], last],
        ).to_arrow_reader(batch_size=25_000)
        rows = (r for batch in reader for r in batch.to_pylist())
        groups = iter(
            (list(g) for _, g in groupby(rows, key=lambda r: (r["timestamp"], r["channel_id"])))
        )
        # Grouping across Arrow batch boundaries is essential; do not break a second.
        group = next(groups, None)
        stream = SparseAdmissionStream()
        expected = {(s["channel_id"], s["prediction_time"]): s for s in samples}
        checked = []
        for at in sorted({s["prediction_time"] for s in samples}):
            while group is not None and group[0]["timestamp"] <= at:
                stream.observe_records(group)
                group = next(groups, None)
            at_channels = sorted({s["channel_id"] for s in samples if s["prediction_time"] == at})
            for actual in stream.evaluate(at, at_channels):
                reference = expected[actual["channel_id"], at]
                for name in (
                    "sensor_type",
                    "admission_status",
                    "admission_reasons",
                    "last_explicit_normal_at",
                    "blocking_qa_count_24h",
                ):
                    if actual[name] != reference[name]:
                        raise ValueError(
                            f"raw-stream/full-batch mismatch {name}: {actual['channel_id']}@{at}"
                        )
                feature_row = sampled_features.get((actual["channel_id"], at))
                if actual["admission_status"] == "eligible":
                    if feature_row is None:
                        raise ValueError("eligible raw-stream check lacks exported features")
                    state = stream._past.channels[actual["channel_id"]]
                    qa = qa_window_counts(state.events, at)
                    for name in QA_NAMES:
                        if feature_row[name] != qa[name]:
                            raise ValueError(f"raw-stream/exported QA feature mismatch: {name}")
                checked.append(
                    {
                        "channel_id": actual["channel_id"],
                        "prediction_time": at.isoformat(),
                        "mismatches": 0,
                        "qa_features_checked": len(QA_NAMES) if feature_row is not None else 0,
                    }
                )
    result = {
        "status": "full_export_and_independent_M1_stream_checks_passed",
        "source_manifest_sha256": sha256(package / "manifest.json"),
        "months": month_results,
        "decision_rows_checked": sum(r["decision_rows"] for r in month_results),
        "feature_rows_checked": sum(r["feature_rows"] for r in month_results),
        "raw_stream_samples": checked,
        "raw_stream_channels": len(channels),
        "raw_observations_replayed": stream.accepted_rows,
        "mismatches": 0,
        "code_lf_sha256": frozen_rule_sha256(Path(__file__)),
    }
    write_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "m1-dir", "a3-dir", "b3-dir", "corrections-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--follow-build", action="store_true",
                   help="verify completed monthly outputs while the local builder runs")
    a = p.parse_args()
    r = verify(
        package=a.package, m1_dir=a.m1_dir, a3_dir=a.a3_dir, b3_dir=a.b3_dir,
        corrections_dir=a.corrections_dir, output=a.output, follow_build=a.follow_build
    )
    print(
        {
            k: r[k]
            for k in (
                "status",
                "decision_rows_checked",
                "feature_rows_checked",
                "raw_stream_channels",
                "mismatches",
            )
        }
    )


if __name__ == "__main__":
    main()
