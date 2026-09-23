"""Optional, disk-partitioned replacement for ``sql.classify``.

The ingestion pipeline selects this fallback with ``--partitioned-classification``.
It uses bounded-fanout Parquet writers and a grouped minimum per hash bucket,
instead of a window over the entire raw table. The caller must supply a new,
empty scratch path outside the published output with enough free space for
another copy of ``raw``. Scratch files are never deleted here.
"""

from pathlib import Path

from stage1.ingestion.schemas import RAW_COLUMNS, STAGING_SCHEMA
from stage1.ingestion.sql import literal


def classify_partitioned(
    connection,
    scratch_dir: str | Path,
    *,
    object_mapping_available: bool = True,
    bucket_count: int = 256,
    writer_fanout: int = 16,
) -> None:
    """Build the same ``classified``, ``conflicts`` and ``clean`` as ``classify``.

    An identical six-column raw key always has the same DuckDB hash and thus
    appears in one bucket, regardless of source, source row, or event year.
    ``min(row_id)`` is therefore the global minimum for that key.  The hash
    need only be stable *within this run*; bucket files are not portable across
    DuckDB versions or intended as a public artifact.

    At most ``writer_fanout`` Parquet partitions are open per COPY. Grouping
    before joining also handles a very frequent raw key without buffering all
    its occurrences in a window. The final conflicts aggregation and clean
    joins retain the original SQL semantics; this fallback specifically bounds
    the expensive deduplication phase.
    """
    if not isinstance(bucket_count, int) or isinstance(bucket_count, bool) or bucket_count < 1:
        raise ValueError("bucket_count must be a positive integer")
    if (
        not isinstance(writer_fanout, int)
        or isinstance(writer_fanout, bool)
        or writer_fanout < 1
        or writer_fanout > 32
    ):
        raise ValueError("writer_fanout must be between 1 and 32")
    scratch = Path(scratch_dir).resolve()
    if scratch.exists():
        raise FileExistsError(f"Scratch path already exists: {scratch}")
    for table in ("classified", "conflicts", "clean"):
        if connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()[0]:
            raise ValueError(f"Derived table already exists: {table}")

    scratch.mkdir(parents=True)
    keys = ", ".join(RAW_COLUMNS)
    names = ", ".join(field.name for field in STAGING_SCHEMA)
    qualified_names = ", ".join(f"r.{field.name}" for field in STAGING_SCHEMA)
    key_join = " AND ".join(f"r.{key} IS NOT DISTINCT FROM f.{key}" for key in RAW_COLUMNS)
    bucket = f"(hash({keys}) % {bucket_count})"
    connection.execute(
        "CREATE TABLE classified AS SELECT *, NULL::BIGINT AS first_row_id, "
        "NULL::VARCHAR AS disposition FROM raw WHERE false"
    )
    for first in range(0, bucket_count, writer_fanout):
        last = min(first + writer_fanout, bucket_count) - 1
        group = scratch / f"group_{first}_{last}"
        connection.execute(
            f"COPY (SELECT {names}, {bucket}::INTEGER AS bucket FROM raw "
            f"WHERE {bucket} BETWEEN {first} AND {last}) "
            f"TO {literal(group.as_posix())} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (bucket), ROW_GROUP_SIZE 2048)"
        )
        for number in range(first, last + 1):
            files = group / f"bucket={number}" / "*.parquet"
            if not files.parent.exists():
                continue
            # Hive partition columns are deliberately disabled. Only the
            # original raw fields enter the classified table.
            connection.execute(f"""
                CREATE TEMP TABLE bucket_first AS
                SELECT {keys}, min(row_id) AS first_row_id
                FROM read_parquet({literal(files.as_posix())}, hive_partitioning=false)
                GROUP BY {keys}
            """)
            connection.execute(f"""
                INSERT INTO classified
                SELECT {qualified_names}, f.first_row_id,
                       CASE WHEN r.repeated_header THEN 'repeated_header'
                            WHEN r.row_id <> f.first_row_id THEN 'exact_duplicate'
                            WHEN r.invalid THEN 'quarantine' ELSE 'accepted' END
                FROM read_parquet({literal(files.as_posix())}, hive_partitioning=false) r
                JOIN bucket_first f ON {key_join}
            """)
            connection.execute("DROP TABLE bucket_first")
    raw_count = connection.execute("SELECT count(*) FROM raw").fetchone()[0]
    classified_count = connection.execute("SELECT count(*) FROM classified").fetchone()[0]
    if classified_count != raw_count:
        raise ValueError(f"Partitioned classification lost rows: {classified_count} != {raw_count}")

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
