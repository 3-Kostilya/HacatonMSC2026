from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from app.config import INCOMING_DIR
from app.dependencies import get_store
from app.raw_pipeline import (
    RAW_ALLOWED_SUFFIXES,
    REFERENCE_ALLOWED_SUFFIXES,
    bootstrap_runtime_assets,
    create_batch,
    get_status,
    process_batch,
)
from app.storage import ParquetStore

router = APIRouter(prefix="/api/raw", tags=["Raw data"])


async def _save(file: UploadFile, directory: Path, allowed: set[str]) -> Path:
    original = Path(file.filename or "upload").name
    suffix = Path(original).suffix.casefold()
    if suffix not in allowed:
        raise ValueError(f"Неподдерживаемый формат {suffix or '(без расширения)'}")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid4().hex}-{original}"
    with path.open("wb") as target:
        while chunk := await file.read(1024 * 1024):
            target.write(chunk)
    return path


@router.get("/capabilities")
def capabilities():
    state = bootstrap_runtime_assets()
    return {
        **state,
        "acceptedJournalFormats": [".csv", ".7z"],
        "acceptedReferenceFormats": [".csv"],
    }


@router.post("/import", status_code=202)
async def import_raw_data(
    background_tasks: BackgroundTasks,
    journal: UploadFile = File(...),
    channels: UploadFile | None = File(default=None),
    objects: UploadFile | None = File(default=None),
    store: ParquetStore = Depends(get_store),
):
    directory = INCOMING_DIR / "raw" / uuid4().hex
    try:
        journal_path = await _save(journal, directory, RAW_ALLOWED_SUFFIXES)
        channels_path = await _save(channels, directory, REFERENCE_ALLOWED_SUFFIXES) if channels else None
        objects_path = await _save(objects, directory, REFERENCE_ALLOWED_SUFFIXES) if objects else None
        if (channels_path is None) != (objects_path is None):
            raise ValueError("Справочник каналов и справочник объектов загружаются вместе")

        batch_id = create_batch(
            journal_path,
            Path(journal.filename or "journal").name,
            channels_path,
            objects_path,
        )
        background_tasks.add_task(
            process_batch,
            store,
            batch_id,
            journal_path,
            Path(journal.filename or "journal").name,
            channels_path,
            objects_path,
        )
        return {"batchId": batch_id, "status": "queued"}
    except ValueError as error:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Не удалось принять файлы: {error}") from error


@router.get("/import/{batch_id}")
def raw_import_status(batch_id: str):
    status = get_status(batch_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Загрузка не найдена")
    return status
