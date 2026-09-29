"""Compare recovered M1 DuckDB tables with their published Parquet exports.

The comparison is a logical multiset comparison: row order and Parquet row-group
layout do not matter.  Every exported column participates in two independently
salted DuckDB row hashes.  Count, 128-bit sum, XOR, minimum, and maximum are
aggregated for both hashes.  Clean data is compared per calendar partition in a
single scan of each side; the two auxiliary exports are compared as whole tables.

DuckDB does not promise that ``hash`` is stable between releases.  Both sides are
therefore read by the same connection and the exact DuckDB version is recorded in
the report.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile

import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1.ingestion.sql import AUXILIARY_EXPORTS  # noqa: E402

ALGORITHM_NAME = "duckdb-logical-multiset-v1"
HASH_SALTS = ("m1-db-parquet/v1/a", "m1-db-parquet/v1/b")
AGGREGATES = ("count", "sum", "bit_xor", "minimum", "maximum")


class FingerprintMismatchError(ValueError):
    """Raised after a complete comparison report has identified a mismatch."""

    def __init__(self, report: dict):
        super().__init__("DuckDB and published Parquet logical fingerprints differ")
        self.report = report


def _identifier(value: str) -> str:
    """Return one safely quoted DuckDB identifier."""
    return '"' + value.replace('"', '""') + '"'


def _literal(value: str | Path) -> str:
    """Return one safely quoted DuckDB string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _parquet_relation(paths: list[Path], *, hive_partitioning: bool = False) -> str:
    if not paths:
        raise ValueError("at least one Parquet path is required")
    arguments = ", ".join(_literal(path.resolve().as_posix()) for path in paths)
    source = arguments if len(paths) == 1 else f"[{arguments}]"
    hive = "true" if hive_partitioning else "false"
    return f"read_parquet({source}, hive_partitioning={hive}, union_by_name=false)"


def _unordered_auxiliary_query(name: str) -> str:
    """Drop only the reviewed export's final presentation ordering.

    Sorting is irrelevant to a multiset fingerprint and could otherwise create a
    very large spill.  Requiring the known final clause makes SQL drift fail closed.
    """
    query = AUXILIARY_EXPORTS[name]
    body, separator, ordering = query.rpartition(" ORDER BY ")
    if not separator or not body or not ordering:
        raise ValueError(f"auxiliary export query has no final ORDER BY: {name}")
    return body


def _describe(connection, query: str) -> list[dict[str, str]]:
    rows = connection.execute(f"DESCRIBE SELECT * FROM ({query}) AS described").fetchall()
    return [{"name": row[0], "type": row[1]} for row in rows]


def _logical_projections(database_schema, parquet_schema):
    """Build equal typed projections for DuckDB's documented export coercions.

    Parquet has no signed 128-bit integer.  DuckDB consequently materializes a
    HUGEINT aggregate such as ``sum(integer)`` as DOUBLE during COPY.  That is an
    expected logical export conversion, not corruption.  No other type change is
    accepted here.
    """
    if parquet_schema is None or len(database_schema) != len(parquet_schema):
        return None
    database_fields = []
    parquet_fields = []
    casts = []
    for database_field, parquet_field in zip(database_schema, parquet_schema):
        if database_field["name"] != parquet_field["name"]:
            return None
        name = database_field["name"]
        identifier = _identifier(name)
        if database_field["type"] == parquet_field["type"]:
            database_fields.append(identifier)
        elif (database_field["type"], parquet_field["type"]) == ("HUGEINT", "DOUBLE"):
            database_fields.append(f"cast({identifier} AS DOUBLE) AS {identifier}")
            casts.append(
                {
                    "column": name,
                    "database_type": "HUGEINT",
                    "parquet_type": "DOUBLE",
                }
            )
        else:
            return None
        parquet_fields.append(identifier)
    return database_fields, parquet_fields, casts


def _hash_projection(columns: list[str]) -> str:
    if not columns:
        raise ValueError("cannot fingerprint a relation without columns")
    values = ", ".join(_identifier(column) for column in columns)
    return ", ".join(
        f"hash({_literal(salt)}, row({values})) AS hash_{index}"
        for index, salt in enumerate(HASH_SALTS)
    )


def _aggregate_projection() -> str:
    fields = ["count(*)::UBIGINT AS row_count"]
    for index in range(len(HASH_SALTS)):
        name = f"hash_{index}"
        fields.extend(
            (
                f"coalesce(sum({name}::HUGEINT), 0)::VARCHAR AS sum_{index}",
                f"coalesce(bit_xor({name}), 0)::VARCHAR AS xor_{index}",
                f"coalesce(min({name}), 0)::VARCHAR AS min_{index}",
                f"coalesce(max({name}), 0)::VARCHAR AS max_{index}",
            )
        )
    return ", ".join(fields)


def _digest(row: tuple, offset: int = 0) -> dict:
    position = offset
    result = {"rows": int(row[position])}
    position += 1
    for index in range(len(HASH_SALTS)):
        total, xor, minimum, maximum = (int(row[position + item]) for item in range(4))
        position += 4
        result[f"hash_{index}"] = {
            "sum_decimal": str(total),
            "xor_hex": f"{xor:016x}",
            "minimum_hex": f"{minimum:016x}",
            "maximum_hex": f"{maximum:016x}",
        }
    return result


def _fingerprint_relation(connection, query: str, columns: list[str]) -> dict:
    sql = f"""
        WITH source_rows AS ({query}),
             row_hashes AS (
                 SELECT {_hash_projection(columns)} FROM source_rows
             )
        SELECT {_aggregate_projection()} FROM row_hashes
    """
    return _digest(connection.execute(sql).fetchone())


def _fingerprint_partitions(
    connection, query: str, columns: list[str]
) -> dict[tuple[int, int], dict]:
    sql = f"""
        WITH source_rows AS ({query}),
             row_hashes AS (
                 SELECT year, month, {_hash_projection(columns)} FROM source_rows
             )
        SELECT year, month, {_aggregate_projection()}
        FROM row_hashes
        GROUP BY year, month
        ORDER BY year, month
    """
    rows = connection.execute(sql).fetchall()
    return {(int(row[0]), int(row[1])): _digest(row, 2) for row in rows}


def _count_partitions(connection, query: str) -> dict[tuple[int, int], int]:
    rows = connection.execute(
        f"""SELECT year, month, count(*) FROM ({query}) AS source_rows
            GROUP BY year, month ORDER BY year, month"""
    ).fetchall()
    return {(int(year), int(month)): int(count) for year, month, count in rows}


def _clean_files(directory: Path) -> tuple[dict[tuple[int, int], list[Path]], list[str]]:
    root = directory / "clean"
    groups: dict[tuple[int, int], list[Path]] = {}
    invalid = []
    if not root.exists():
        return groups, invalid
    for path in sorted(root.rglob("*.parquet")):
        relative = path.relative_to(root)
        parts = relative.parts
        try:
            if (
                len(parts) != 3
                or not parts[0].startswith("year=")
                or not parts[1].startswith("month=")
            ):
                raise ValueError
            year = int(parts[0].split("=", 1)[1])
            month = int(parts[1].split("=", 1)[1])
            if not 1 <= month <= 12:
                raise ValueError
        except ValueError:
            invalid.append(relative.as_posix())
            continue
        groups.setdefault((year, month), []).append(path)
    return groups, invalid


def _side(query: str | None, connection, columns: list[str] | None) -> dict | None:
    if query is None:
        return None
    if columns is None:
        rows = connection.execute(f"SELECT count(*) FROM ({query}) AS counted").fetchone()[0]
        return {"rows": int(rows), "digest": None}
    fingerprint = _fingerprint_relation(connection, query, columns)
    return {
        "rows": fingerprint["rows"],
        "digest": {key: value for key, value in fingerprint.items() if key != "rows"},
    }


def _compare_auxiliary(connection, database_query: str, parquet_path: Path) -> dict:
    parquet_query = None
    if parquet_path.is_file():
        parquet_query = f"SELECT * FROM {_parquet_relation([parquet_path])}"
    database_schema = _describe(connection, database_query)
    parquet_schema = _describe(connection, parquet_query) if parquet_query is not None else None
    projections = _logical_projections(database_schema, parquet_schema)
    compatible_schema = projections is not None
    casts = []
    if compatible_schema:
        database_fields, parquet_fields, casts = projections
        columns = [item["name"] for item in database_schema]
        database_query = (
            f"SELECT {', '.join(database_fields)} FROM ({database_query}) AS database_export"
        )
        parquet_query = (
            f"SELECT {', '.join(parquet_fields)} FROM ({parquet_query}) AS parquet_export"
        )
    else:
        columns = None
    database_side = _side(database_query, connection, columns)
    parquet_side = _side(parquet_query, connection, columns)
    match = compatible_schema and database_side["digest"] == parquet_side["digest"]
    result = {
        "match": match,
        "schema": {"database": database_schema, "parquet": parquet_schema},
        "logical_export_casts": casts,
        "database": database_side,
        "parquet": parquet_side,
    }
    if not compatible_schema:
        result["reason"] = "missing_parquet" if parquet_query is None else "schema_mismatch"
    elif not match:
        result["reason"] = "logical_content_mismatch"
    return result


def _compare_clean(connection, artifact: Path) -> dict:
    files, invalid = _clean_files(artifact)
    all_files = [path for paths in files.values() for path in paths]
    database_schema = _describe(connection, "SELECT * EXCLUDE (year, month) FROM clean")
    parquet_schema = None
    parquet_relation = None
    if all_files:
        parquet_relation = _parquet_relation(all_files, hive_partitioning=True)
        parquet_schema = _describe(
            connection, f"SELECT * EXCLUDE (year, month) FROM {parquet_relation}"
        )
    same_schema = database_schema == parquet_schema
    columns = [item["name"] for item in database_schema]
    projection = ", ".join(_identifier(column) for column in columns)
    database_query = f"SELECT year, month, {projection} FROM clean"
    parquet_query = (
        f"SELECT year::INTEGER AS year, month::INTEGER AS month, {projection} "
        f"FROM {parquet_relation}"
        if same_schema
        else None
    )

    if same_schema:
        database = _fingerprint_partitions(connection, database_query, columns)
        parquet = _fingerprint_partitions(connection, parquet_query, columns)
    else:
        database = {
            key: {"rows": value, "digest": None}
            for key, value in _count_partitions(connection, "SELECT year, month FROM clean").items()
        }
        parquet = {}
        if parquet_relation is not None:
            parquet = {
                key: {"rows": value, "digest": None}
                for key, value in _count_partitions(
                    connection,
                    f"SELECT year::INTEGER AS year, month::INTEGER AS month "
                    f"FROM {parquet_relation}",
                ).items()
            }

    partitions = []
    for year, month in sorted(set(database) | set(parquet)):
        database_digest = database.get((year, month))
        parquet_digest = parquet.get((year, month))
        match = same_schema and database_digest == parquet_digest
        partition = {
            "year": year,
            "month": month,
            "match": match,
            "database": database_digest,
            "parquet": parquet_digest,
        }
        if not match:
            if database_digest is None:
                partition["reason"] = "unexpected_parquet_partition"
            elif parquet_digest is None:
                partition["reason"] = "missing_parquet_partition"
            elif not same_schema:
                partition["reason"] = "schema_mismatch"
            else:
                partition["reason"] = "logical_content_mismatch"
        partitions.append(partition)
    return {
        "match": same_schema and not invalid and all(item["match"] for item in partitions),
        "schema": {"database": database_schema, "parquet": parquet_schema},
        "invalid_parquet_paths": invalid,
        "partitions": partitions,
    }


def compare_db_to_parquet(
    database: Path,
    artifact: Path,
    *,
    memory_limit: str = "1GB",
    threads: int = 2,
    temp_directory: Path | None = None,
) -> dict:
    """Return a full report, or raise with that report when any logical data differs."""
    database = database.resolve()
    artifact = artifact.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    if not artifact.is_dir():
        raise NotADirectoryError(artifact)
    if threads < 1:
        raise ValueError("threads must be positive")

    temporary = (
        nullcontext(Path(temp_directory).resolve())
        if temp_directory is not None
        else tempfile.TemporaryDirectory(prefix="m1-fingerprint-")
    )
    with temporary as temporary_value:
        scratch = Path(temporary_value)
        scratch.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(
            str(database),
            read_only=True,
            config={
                "memory_limit": memory_limit,
                "threads": str(threads),
                "temp_directory": str(scratch),
                "preserve_insertion_order": "false",
            },
        )
        try:
            tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
            required = {"raw", "classified", "clean"}
            if not required.issubset(tables):
                missing = sorted(required - tables)
                raise ValueError(f"database is missing required derived tables: {missing}")
            clean = _compare_clean(connection, artifact)
            excluded = _compare_auxiliary(
                connection,
                _unordered_auxiliary_query("excluded_rows"),
                artifact / "excluded_rows.parquet",
            )
            statistics = _compare_auxiliary(
                connection,
                _unordered_auxiliary_query("sensor_statistics"),
                artifact / "sensor_statistics.parquet",
            )
            effective_memory = connection.execute(
                "SELECT current_setting('memory_limit')"
            ).fetchone()[0]
        finally:
            connection.close()

    matches = clean["match"] and excluded["match"] and statistics["match"]
    report = {
        "status": "verified" if matches else "mismatch",
        "algorithm": {
            "name": ALGORITHM_NAME,
            "duckdb_version": duckdb.__version__,
            "hash_function": "duckdb hash(salt, row(all_exported_columns))",
            "salts": list(HASH_SALTS),
            "aggregates": list(AGGREGATES),
            "order_independent": True,
        },
        "runtime": {
            "memory_limit": effective_memory,
            "threads": threads,
        },
        "database": str(database),
        "artifact": str(artifact),
        "clean": clean,
        "excluded_rows": excluded,
        "sensor_statistics": statistics,
    }
    if not matches:
        raise FingerprintMismatchError(report)
    return report


def _write_report(report: dict, path: Path | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if path is not None:
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--temp-directory", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = compare_db_to_parquet(
            args.database,
            args.artifact,
            memory_limit=args.memory_limit,
            threads=args.threads,
            temp_directory=args.temp_directory,
        )
    except FingerprintMismatchError as error:
        _write_report(error.report, args.report)
        return 1
    _write_report(report, args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
