"""Disk-backed global deduplication and cardinality-preserving dictionary joins."""

from stage1.ingestion.schemas import RAW_COLUMNS


def literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def classify(connection, *, object_mapping_available=True) -> None:
    keys = ", ".join(RAW_COLUMNS)
    connection.execute(f"""
        CREATE TABLE classified AS
        SELECT *, CASE WHEN repeated_header THEN 'repeated_header'
                       WHEN row_id <> first_row_id THEN 'exact_duplicate'
                       WHEN invalid THEN 'quarantine' ELSE 'accepted' END AS disposition
        FROM (SELECT *, min(row_id) OVER (PARTITION BY {keys}) AS first_row_id FROM raw)
    """)
    connection.execute("""
        CREATE TABLE conflicts AS
        SELECT channel_id, timestamp
        FROM classified WHERE disposition='accepted'
        GROUP BY channel_id, timestamp
        HAVING count(DISTINCT (value_raw, alarm)) > 1
    """)
    missing_mapping_status = (
        "object_id_missing" if object_mapping_available else "object_mapping_unavailable"
    )
    connection.execute(f"""
        CREATE TABLE clean AS
        SELECT r.* EXCLUDE (invalid, repeated_header, first_row_id, disposition, quality_flags),
               c.sensor_type, c.engineering_system_type, c.engineering_system_tag, c.sensor_name,
               c.object_id, o.hierarchy_level, o.parent_object_id, o.object_kind, o.object_name,
               CASE WHEN c.channel_id IS NULL THEN 'unknown_channel'
                    WHEN c.object_id IS NULL THEN '{missing_mapping_status}'
                    WHEN o.object_id IS NULL THEN 'object_not_found' ELSE 'linked' END AS join_status,
               list_concat(r.quality_flags,
                    CASE WHEN c.channel_id IS NULL THEN ['unknown_channel'] ELSE [] END,
                    CASE WHEN x.channel_id IS NOT NULL THEN ['channel_time_conflict'] ELSE [] END
               ) AS quality_flags,
               year(r.timestamp)::INTEGER AS year, month(r.timestamp)::INTEGER AS month
        FROM classified r
        LEFT JOIN channels c USING(channel_id)
        LEFT JOIN objects o ON c.object_id=o.object_id
        LEFT JOIN conflicts x ON r.channel_id=x.channel_id AND r.timestamp=x.timestamp
        WHERE r.disposition='accepted'
    """)


def export_tables(connection, output) -> None:
    clean_dir = output / "clean"
    # A partitioned COPY can flush row groups out of order even with ORDER BY.
    # Export each month through a single ordered writer instead.
    partitions = connection.execute(
        "SELECT DISTINCT year, month FROM clean ORDER BY year, month"
    ).fetchall()
    for year, month in partitions:
        partition = clean_dir / f"year={year}" / f"month={month}"
        partition.mkdir(parents=True, exist_ok=False)
        connection.execute(f"""COPY (
            SELECT * EXCLUDE (year, month) FROM clean
            WHERE year={int(year)} AND month={int(month)}
            ORDER BY channel_id, timestamp, row_id
        ) TO {literal((partition / "data_0.parquet").as_posix())}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)""")
    for name, query in {
        "excluded_rows": """SELECT r.*, f.source AS duplicate_of_source,
            f.source_row AS duplicate_of_source_row FROM classified r
            LEFT JOIN raw f ON r.first_row_id=f.row_id AND r.disposition='exact_duplicate'
            WHERE r.disposition<>'accepted' ORDER BY r.row_id""",
        "sensor_statistics": """SELECT channel_id, sensor_type, count(*) AS rows,
            count(value_numeric) AS numeric_count, count(value_state) AS state_count,
            sum(alarm::INTEGER) AS alarm_count, min(timestamp) AS first_at,
            max(timestamp) AS last_at, count(DISTINCT value_raw) AS distinct_values
            FROM clean GROUP BY channel_id, sensor_type ORDER BY channel_id""",
    }.items():
        connection.execute(
            f"COPY ({query}) TO {literal((output / (name + '.parquet')).as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
