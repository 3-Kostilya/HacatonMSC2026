from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import UploadFile

from app.config import (
    FAILED_DIR,
    INCOMING_DIR,
    PROCESSED_DIR,
)
from app.storage import (
    ParquetStore,
)

ALLOWED_SUFFIXES = {
    ".parquet",
    ".csv",
}


def _now() -> datetime:
    return datetime.now(
        timezone.utc
    ).replace(
        tzinfo=None
    )


def _hash_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as stream:

        for chunk in iter(
            lambda:
                stream.read(
                    1024 * 1024
                ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def _read_upload(
    path: Path,
) -> pd.DataFrame:

    if (
        path.suffix.lower()
        == ".parquet"
    ):
        return pd.read_parquet(
            path,
            engine="pyarrow",
        )

    try:
        return pd.read_csv(
            path,
            low_memory=False,
            encoding="utf-8-sig",
        )

    except UnicodeDecodeError:
        return pd.read_csv(
            path,
            low_memory=False,
            encoding="cp1251",
        )


def detect_kind(
    columns: set[str],
) -> str:

    if {
        "episode_id",
        "channel_id",
        "decision",
        "confirmed_at",
    }.issubset(
        columns
    ):
        return "episodes"

    if {
        "channel_id",
        "prediction_time",
        "prediction_status",
    }.issubset(
        columns
    ):
        return "forecasts"

    event_aliases = set(
        columns
    )

    if (
        "value_state"
        in event_aliases
        or "value_raw"
        in event_aliases
    ):
        event_aliases.add(
            "raw_value"
        )

    if (
        "state"
        in event_aliases
    ):
        event_aliases.add(
            "raw_value"
        )

    if {
        "channel_id",
        "timestamp",
        "raw_value",
        "alarm",
    }.issubset(
        event_aliases
    ):
        return "events"

    if {
        "channel_id",
        "sensor_type",
    }.issubset(
        columns
    ):
        return "sensors"

    return "unknown"


async def save_upload(
    file: UploadFile,
) -> tuple[
    Path,
    str,
]:

    original = Path(
        file.filename
        or "upload"
    ).name

    suffix = (
        Path(
            original
        )
        .suffix
        .lower()
    )

    if (
        suffix
        not in ALLOWED_SUFFIXES
    ):
        raise ValueError(
            "Поддерживаются только "
            ".parquet и .csv"
        )

    stored_name = (
        f"{uuid4().hex}-"
        f"{original}"
    )

    path = (
        INCOMING_DIR
        / stored_name
    )

    with path.open(
        "wb"
    ) as target:

        while chunk := (
            await file.read(
                1024 * 1024
            )
        ):
            target.write(
                chunk
            )

    return (
        path,
        original,
    )


def import_uploaded_file(
    store: ParquetStore,
    path: Path,
    original_filename: str,
    requested_kind: str = "auto",
) -> dict:

    batch_id = uuid4().hex

    file_hash = _hash_file(
        path
    )

    if store.has_file_hash(
        file_hash
    ):

        path.unlink(
            missing_ok=True
        )

        raise FileExistsError(
            "Этот файл уже "
            "был импортирован"
        )

    record = {
        "batch_id":
            batch_id,

        "filename":
            original_filename,

        "stored_filename":
            path.name,

        "file_hash":
            file_hash,

        "dataset_kind":
            requested_kind,

        "rows_count":
            0,

        "status":
            "processing",

        "error_message":
            None,

        "imported_at":
            _now(),

        "finished_at":
            None,
    }

    store.add_import_record(
        record
    )

    try:

        frame = _read_upload(
            path
        )

        if (
            requested_kind
            == "auto"
        ):
            kind = detect_kind(
                set(
                    frame.columns
                )
            )

        else:
            kind = (
                requested_kind
            )

        if kind == "unknown":
            raise ValueError(
                "Не удалось определить "
                "тип данных. "
                "Ожидаются события, "
                "прогнозы, эпизоды "
                "или справочник датчиков."
            )

        if kind == "events":

            rows = (
                store.append_events(
                    frame
                )
            )

        elif kind == "forecasts":

            rows = (
                store.append_forecasts(
                    frame
                )
            )

        elif kind == "episodes":

            rows = (
                store.append_episodes(
                    frame
                )
            )

        elif kind == "sensors":

            rows = (
                store.upsert_sensors(
                    frame
                )
            )

        else:
            raise ValueError(
                "Неизвестный "
                "dataset kind: "
                f"{kind}"
            )

        destination = (
            PROCESSED_DIR
            / path.name
        )

        shutil.move(
            str(path),
            str(destination),
        )

        store.update_import_record(
            batch_id,

            dataset_kind=kind,

            rows_count=rows,

            status="processed",

            stored_filename=(
                destination.name
            ),

            finished_at=_now(),
        )

        return {
            "batchId":
                batch_id,

            "filename":
                original_filename,

            "datasetKind":
                kind,

            "rowsCount":
                rows,

            "status":
                "processed",

            "errorMessage":
                None,
        }

    except Exception as error:

        if path.exists():

            destination = (
                FAILED_DIR
                / path.name
            )

            shutil.move(
                str(path),
                str(destination),
            )

            stored_name = (
                destination.name
            )

        else:
            stored_name = (
                path.name
            )

        store.update_import_record(
            batch_id,

            rows_count=0,

            status="failed",

            stored_filename=(
                stored_name
            ),

            error_message=str(
                error
            ),

            finished_at=_now(),
        )

        raise