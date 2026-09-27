import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)

from app.dependencies import (
    get_store,
)
from app.import_service import (
    import_uploaded_file,
    save_upload,
)
from app.schemas import (
    ImportResponse,
)
from app.storage import (
    ParquetStore,
)

router = APIRouter(
    prefix="/api",
    tags=[
        "Data import",
    ],
)


def _clean(
    value,
):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None

    except (
        TypeError,
        ValueError,
    ):
        pass

    return value


@router.post(
    "/import",
    response_model=ImportResponse,
    status_code=201,
)
async def import_data(
    file: UploadFile = File(...),

    kind: str = Query(
        default="auto",
        pattern=(
            "^(auto|events|"
            "forecasts|episodes|"
            "sensors)$"
        ),
    ),

    store: ParquetStore = Depends(
        get_store
    ),
):

    try:

        path, original = (
            await save_upload(
                file
            )
        )

        return import_uploaded_file(
            store,
            path,
            original,
            kind,
        )

    except FileExistsError as error:

        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Import failed: "
                f"{error}"
            ),
        ) from error


@router.get(
    "/imports"
)
def get_imports(
    store: ParquetStore = Depends(
        get_store
    ),
):

    frame = store.imports()

    if frame.empty:
        return []

    frame = frame.sort_values(
        "imported_at",
        ascending=False,
    )

    result = []

    for _, row in (
        frame.iterrows()
    ):

        rows_count = _clean(
            row.get(
                "rows_count"
            )
        )

        error_message = _clean(
            row.get(
                "error_message"
            )
        )

        result.append(
            {
                "batchId":
                    str(
                        _clean(
                            row.get(
                                "batch_id"
                            )
                        )
                        or ""
                    ),

                "filename":
                    str(
                        _clean(
                            row.get(
                                "filename"
                            )
                        )
                        or ""
                    ),

                "datasetKind":
                    str(
                        _clean(
                            row.get(
                                "dataset_kind"
                            )
                        )
                        or "unknown"
                    ),

                "rowsCount":
                    (
                        int(
                            rows_count
                        )
                        if rows_count
                        is not None
                        else 0
                    ),

                "status":
                    str(
                        _clean(
                            row.get(
                                "status"
                            )
                        )
                        or "unknown"
                    ),

                "errorMessage":
                    (
                        str(
                            error_message
                        )
                        if error_message
                        is not None
                        else None
                    ),
            }
        )

    return result