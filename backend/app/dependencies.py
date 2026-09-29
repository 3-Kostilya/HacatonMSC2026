from app.storage import (
    ParquetStore,
    store,
)


def get_store() -> ParquetStore:
    return store