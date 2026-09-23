"""Crash-safe JSON checkpoints used by the ingestion pipeline."""

import json
import os
from pathlib import Path
import tempfile


def atomic_write_json(path: Path, payload: object, *, default=None) -> None:
    """Durably replace a JSON file without exposing a truncated checkpoint.

    The temporary file is created beside the target, so ``os.replace`` remains a
    single-filesystem atomic operation.  A failed replace leaves the previous
    checkpoint intact; the temporary file created by this call is removed when
    possible.
    """
    path = Path(path)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(payload, stream, ensure_ascii=False, indent=2, default=default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
