from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq

from app.config import (
    CHANNELS_REFERENCE_FILE,
    FAILED_DIR,
    OBJECTS_REFERENCE_FILE,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_RUNS_DIR,
    REFERENCE_DIR,
    RUNTIME_MODEL_DIR,
)
from app.live_inference import score_channels
from app.storage import ParquetStore

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage1.ingestion.dictionaries import load_dictionaries  # noqa: E402
from stage1.ingestion.pipeline import run_ingestion  # noqa: E402
from ml.service_candidate.loader import ResearchRiskModel  # noqa: E402

RAW_ALLOWED_SUFFIXES = {".csv", ".7z"}
REFERENCE_ALLOWED_SUFFIXES = {".csv"}


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status_path(batch_id: str) -> Path:
    return RAW_RUNS_DIR / f"{batch_id}.status.json"


def _write_status(batch_id: str, **values: Any) -> None:
    path = _status_path(batch_id)
    current: dict[str, Any] = {}
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(values)
    current["updatedAt"] = _now().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def get_status(batch_id: str) -> dict[str, Any] | None:
    if re.fullmatch(r"[0-9a-f]{32}", batch_id) is None:
        return None
    path = _status_path(batch_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_runtime_assets() -> dict[str, bool]:
    """Check the pinned runtime model and explicitly supplied references.

    Never discover arbitrary CSVs or model bundles: repository fixtures and
    research artifacts must not silently become operational inputs.
    """

    model_ready = False
    if (RUNTIME_MODEL_DIR / "model.cbm").is_file() and (
        RUNTIME_MODEL_DIR / "model_metadata.json"
    ).is_file():
        try:
            ResearchRiskModel(RUNTIME_MODEL_DIR)
            model_ready = True
        except (ValueError, KeyError, OSError):
            model_ready = False

    return {
        "modelReady": model_ready,
        "referencesReady": CHANNELS_REFERENCE_FILE.is_file()
        and OBJECTS_REFERENCE_FILE.is_file(),
    }


def create_batch(
    journal_path: Path,
    original_filename: str,
    channels_path: Path | None,
    objects_path: Path | None,
) -> str:
    batch_id = uuid4().hex
    _write_status(
        batch_id,
        batchId=batch_id,
        filename=original_filename,
        status="queued",
        stage="Ожидание обработки",
        rowsCount=0,
        forecastsCount=0,
        errorMessage=None,
        journalPath=str(journal_path),
        channelsPath=str(channels_path) if channels_path else None,
        objectsPath=str(objects_path) if objects_path else None,
    )
    return batch_id


def _sensor_frame(channels_path: Path, objects_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    channels, objects, audit = load_dictionaries(channels_path, objects_path)
    objects_by_id = {str(row["object_id"]): row for row in objects}
    rows: list[dict[str, Any]] = []
    for channel in channels:
        object_row = objects_by_id.get(str(channel.get("object_id"))) if channel.get("object_id") else None
        rows.append(
            {
                **channel,
                "name": channel.get("sensor_name") or None,
                "hierarchy_level": object_row.get("hierarchy_level") if object_row else None,
                "parent_object_id": object_row.get("parent_object_id") if object_row else None,
                "object_kind": object_row.get("object_kind") if object_row else None,
                "object_name": object_row.get("object_name") if object_row else None,
                "updated_at": _now(),
            }
        )
    return pd.DataFrame(rows), audit


def _publish_references(channels_path: Path, objects_path: Path) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        (channels_path, CHANNELS_REFERENCE_FILE),
        (objects_path, OBJECTS_REFERENCE_FILE),
    ):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)


def _append_clean_run(store: ParquetStore, run_dir: Path) -> tuple[int, set[str]]:
    total = 0
    channels: set[str] = set()
    for path in sorted((run_dir / "clean").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=50_000):
            frame = batch.to_pandas()
            if "channel_id" in frame.columns:
                channels.update(str(value) for value in frame["channel_id"].dropna().astype(str))
            total += store.append_events(frame)
    return total, channels


def process_batch(
    store: ParquetStore,
    batch_id: str,
    journal_path: Path,
    original_filename: str,
    channels_path: Path | None,
    objects_path: Path | None,
) -> None:
    run_dir = RAW_RUNS_DIR / batch_id
    import_record_created = False
    try:
        _write_status(batch_id, status="processing", stage="Проверка файла")
        journal_hash = _sha256(journal_path)
        if store.has_file_hash(journal_hash):
            raise FileExistsError("Этот журнал уже был импортирован")
        store.add_import_record(
            {
                "batch_id": batch_id,
                "filename": original_filename,
                "stored_filename": journal_path.name,
                "file_hash": journal_hash,
                "dataset_kind": "raw_events",
                "rows_count": 0,
                "status": "processing",
                "error_message": None,
                "imported_at": _now(),
                "finished_at": None,
            }
        )
        import_record_created = True
        _write_status(batch_id, stage="Проверка справочников")
        bootstrap_runtime_assets()

        selected_channels = channels_path or CHANNELS_REFERENCE_FILE
        selected_objects = objects_path or OBJECTS_REFERENCE_FILE
        if not selected_channels.is_file() or not selected_objects.is_file():
            raise ValueError(
                "Для первой загрузки нужны справочник каналов и справочник объектов. "
                "После успешной загрузки они сохраняются в backend/data/reference."
            )

        sensor_frame, dictionary_audit = _sensor_frame(selected_channels, selected_objects)
        _write_status(batch_id, stage="Нормализация сырого журнала")

        config = {
            "channels": str(selected_channels.resolve()),
            "objects": str(selected_objects.resolve()),
            "output": str(run_dir.resolve()),
            "batch_size": 25_000,
            "memory_limit": "1GB",
            "sources": [
                {
                    "path": str(journal_path.resolve()),
                    "max_rows": None,
                }
            ],
        }
        run_ingestion(config)

        _write_status(batch_id, stage="Сохранение событий в backend")
        store.upsert_sensors(sensor_frame)
        rows, affected_channels = _append_clean_run(store, run_dir)

        if channels_path is not None or objects_path is not None:
            _publish_references(selected_channels, selected_objects)

        _write_status(batch_id, stage="Расчёт признаков и прогноза", rowsCount=rows)
        forecasts = score_channels(store, affected_channels)

        destination = PROCESSED_DIR / "raw" / batch_id
        destination.mkdir(parents=True, exist_ok=True)
        for path in (journal_path, channels_path, objects_path):
            if path is not None and path.exists():
                shutil.move(str(path), str(destination / path.name))

        _write_status(
            batch_id,
            status="processed",
            stage="Готово",
            rowsCount=rows,
            forecastsCount=forecasts,
            sensorsCount=len(affected_channels),
            dictionaryAudit=dictionary_audit,
            processedPath=str(destination),
            runPath=str(run_dir),
        )

        store.update_import_record(
            batch_id,
            stored_filename=str(destination / journal_path.name),
            rows_count=rows,
            status="processed",
            error_message=None,
            finished_at=_now(),
        )

    except Exception as error:
        failed_dir = FAILED_DIR / "raw" / batch_id
        failed_dir.mkdir(parents=True, exist_ok=True)
        for path in (journal_path, channels_path, objects_path):
            try:
                if path is not None and path.exists():
                    shutil.move(str(path), str(failed_dir / path.name))
            except Exception:
                pass
        if import_record_created:
            store.update_import_record(
                batch_id,
                rows_count=0,
                status="failed",
                error_message=str(error),
                finished_at=_now(),
            )
        _write_status(
            batch_id,
            status="failed",
            stage="Ошибка",
            errorMessage=str(error),
            traceback=traceback.format_exc(limit=8),
        )
