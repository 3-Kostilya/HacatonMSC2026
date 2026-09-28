"""Stream ingestion-v1 clean Parquet as the agreed ml-m0-v1 clean envelope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from stage1.ml_m0_contracts import CONTRACT_VERSION, SCHEMAS


def iter_m0_clean_batches(directory: Path, batch_size: int = 25_000) -> Iterator[pa.RecordBatch]:
    """Project clean rows without loading the full history or rewriting it."""
    directory = Path(directory).resolve()
    if directory.name.endswith(".inprogress"):
        raise ValueError("only published ingestion artifacts can be projected")
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["status"] != "complete" or manifest["schema_version"] != "ingestion-v1":
        raise ValueError("a complete ingestion-v1 artifact is required")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    config_bytes = json.dumps(
        manifest["configuration"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    provenance = {
        "schema_version": CONTRACT_VERSION,
        "run_id": f"ingestion-{manifest_hash[:16]}",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "input_manifest_sha256": manifest_hash,
    }
    schema = SCHEMAS["clean"]
    data_names = schema.names[4:]
    for path in sorted((directory / "clean").rglob("*.parquet")):
        file = pq.ParquetFile(path)
        if not set(data_names).issubset(file.schema_arrow.names):
            raise ValueError(f"clean Parquet lacks M0 fields: {path}")
        for batch in file.iter_batches(batch_size=batch_size, columns=data_names):
            columns = [
                pa.array([provenance[name]] * batch.num_rows, type=schema.field(name).type)
                for name in schema.names[:4]
            ]
            columns.extend(batch.column(batch.schema.get_field_index(name)) for name in data_names)
            yield pa.RecordBatch.from_arrays(columns, schema=schema)
