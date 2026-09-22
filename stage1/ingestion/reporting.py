"""Small aggregate reports only; event history never materializes in Python."""

import csv
import json


def rows(connection, query):
    result = connection.execute(query)
    names = [item[0] for item in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


def build_quality_report(connection, output, manifest, dictionary_audit):
    totals = dict(
        connection.execute(
            "SELECT disposition, count(*) FROM classified GROUP BY disposition"
        ).fetchall()
    )
    accepted = totals.get("accepted", 0)
    report = {
        "schema_version": "ingestion-v1",
        "scope": manifest["scope"],
        "sources": manifest["sources"],
        "input_rows": manifest["input_rows"],
        "dispositions": totals,
        "dictionary_audit": dictionary_audit,
        "by_type": rows(
            connection,
            """SELECT coalesce(sensor_type,'unknown') AS sensor_type,
            count(*) AS rows, count(DISTINCT channel_id) AS channels,
            avg(is_numeric::INTEGER) AS numeric_fraction, avg(alarm::INTEGER) AS alarm_fraction,
            min(timestamp) AS first_at, max(timestamp) AS last_at,
            date_diff('second',min(timestamp),max(timestamp)) AS history_span_seconds
            FROM clean GROUP BY sensor_type ORDER BY rows DESC""",
        ),
        "by_partition": rows(
            connection,
            "SELECT year, month, count(*) AS rows FROM clean GROUP BY year, month ORDER BY year, month",
        ),
        "join_status": rows(
            connection, "SELECT join_status, count(*) AS rows FROM clean GROUP BY join_status"
        ),
        "quality_flags": rows(
            connection,
            "SELECT flag, count(*) AS rows FROM (SELECT unnest(quality_flags) AS flag FROM clean) GROUP BY flag",
        ),
        "repeated_event_ids": connection.execute("""SELECT coalesce(sum(n-1),0) FROM
            (SELECT count(*) n FROM clean WHERE event_id IS NOT NULL GROUP BY event_id HAVING count(*)>1)""").fetchone()[
            0
        ],
        "special_numeric_values": rows(
            connection,
            "SELECT value_raw, count(*) AS rows FROM clean WHERE value_numeric IN (-3276,327.68,999) GROUP BY value_raw",
        ),
        "sanity_checks": {
            "row_conservation": sum(totals.values()) == manifest["input_rows"],
            "joins_do_not_multiply_rows": connection.execute(
                "SELECT count(*) FROM clean"
            ).fetchone()[0]
            == accepted,
            "value_representation_consistent": connection.execute("""SELECT count(*) FROM clean WHERE
                is_numeric <> (value_numeric IS NOT NULL) OR
                (is_numeric AND value_state IS NOT NULL) OR (NOT is_numeric AND value_state IS NULL)""").fetchone()[
                0
            ]
            == 0,
            "no_duplicate_source_row": connection.execute(
                "SELECT count(*)-count(DISTINCT (source,source_row)) FROM clean"
            ).fetchone()[0]
            == 0,
        },
    }
    if not all(report["sanity_checks"].values()):
        raise ValueError(f"Ingestion sanity checks failed: {report['sanity_checks']}")
    # No candidate counts: this milestone does not run anomaly detection.
    family_columns = [
        "sensor_type",
        "rows",
        "channels",
        "numeric_fraction",
        "alarm_fraction",
        "first_at",
        "last_at",
        "history_span_seconds",
    ]
    with (output / "sensor_family_report.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=family_columns)
        writer.writeheader()
        writer.writerows(report["by_type"])
    report["output_files"] = [
        {"path": str(p.relative_to(output)), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*.parquet"))
    ]
    report["parquet_bytes"] = sum(item["bytes"] for item in report["output_files"])
    (output / "data_quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report
