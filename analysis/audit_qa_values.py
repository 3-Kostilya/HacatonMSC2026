"""Read-only audit of customer-mentioned values across the published M1 Parquet.

The report counts observations. It does not assign physical-failure labels or
change the published M1, R3, R4, or R5 artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb


SERVICE_VALUE_CANDIDATES = (
    "-3276",
    "-127",
    "-100",
    "-255",
    "255",
    "327.68",
    "999",
    "01.01.1970 03:00:00",
    "01.01.1970 03:00:01",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(
    connection: duckdb.DuckDBPyConnection, sql: str, params: list | None = None
) -> list[dict]:
    result = connection.execute(sql, params or [])
    fields = [item[0] for item in result.description]
    return [dict(zip(fields, row)) for row in result.fetchall()]


def audit(m1_dir: Path) -> dict:
    m1_dir = m1_dir.resolve()
    manifest_path = m1_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("M1 must be published and complete")
    paths = sorted((m1_dir / "clean").rglob("*.parquet"))
    if not paths:
        raise ValueError("M1 has no clean Parquet files")
    if any("year=2021" in path.parts for path in paths):
        raise ValueError("Excluded 2021 archive must not be audited as clean history")

    con = duckdb.connect()
    try:
        con.execute("SET threads=4")
        con.execute("SET memory_limit='2GB'")
        con.read_parquet([str(path) for path in paths], hive_partitioning=True).create_view("clean")
        placeholders = ",".join("?" for _ in SERVICE_VALUE_CANDIDATES)
        service_values = _records(
            con,
            f"""
            SELECT year, sensor_type, value_raw, alarm, count(*) AS rows,
                   count(DISTINCT channel_id) AS channels
            FROM clean
            WHERE value_raw IN ({placeholders})
            GROUP BY year, sensor_type, value_raw, alarm
            ORDER BY year, sensor_type, value_raw, alarm
            """,
            list(SERVICE_VALUE_CANDIDATES),
        )
        gas_numeric = _records(
            con,
            """
            SELECT year, alarm,
                   CASE WHEN value_numeric < 0 THEN 'negative'
                        WHEN value_numeric < 0.1 THEN '[0,0.1)'
                        WHEN value_numeric < 1 THEN '[0.1,1)'
                        WHEN value_numeric < 5 THEN '[1,5)'
                        WHEN value_numeric < 15 THEN '[5,15)'
                        WHEN value_numeric <= 100 THEN '[15,100]'
                        ELSE '>100' END AS value_band,
                   count(*) AS rows, count(DISTINCT channel_id) AS channels,
                   min(value_numeric) AS minimum, max(value_numeric) AS maximum
            FROM clean
            WHERE sensor_type = 'Газовый датчик' AND value_numeric IS NOT NULL
            GROUP BY year, alarm, value_band
            ORDER BY year, alarm, value_band
            """,
        )
        gas_text = _records(
            con,
            """
            SELECT year, value_state, alarm, count(*) AS rows
            FROM clean
            WHERE sensor_type = 'Газовый датчик'
              AND value_state IN ('Обнаружен газ', 'Норма', 'Неисправен')
            GROUP BY year, value_state, alarm
            ORDER BY year, value_state, alarm
            """,
        )
        gas_negative_values = _records(
            con,
            """
            SELECT value_numeric, count(*) AS rows,
                   count(DISTINCT channel_id) AS channels
            FROM clean
            WHERE sensor_type = 'Газовый датчик' AND value_numeric < 0
            GROUP BY value_numeric
            ORDER BY rows DESC, value_numeric
            LIMIT 30
            """,
        )
    finally:
        con.close()

    return {
        "schema_version": "qa-value-audit-v2",
        "source_m1_manifest_sha256": _sha256(manifest_path),
        "source_parquet_files": len(paths),
        "excluded_years": [2021],
        "service_value_candidates": list(SERVICE_VALUE_CANDIDATES),
        "service_values_by_year_type_alarm": service_values,
        "gas_numeric_bands_by_year_alarm": gas_numeric,
        "gas_text_by_year_alarm": gas_text,
        "gas_negative_top_values": gas_negative_values,
        "interpretation": "counts only; technical codes and gas alarms are not physical-failure labels",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.m1_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_parquet_files": report["source_parquet_files"],
                "service_groups": len(report["service_values_by_year_type_alarm"]),
                "gas_numeric_groups": len(report["gas_numeric_bands_by_year_alarm"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
