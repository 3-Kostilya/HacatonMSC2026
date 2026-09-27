"""Compatibility module.

SQL Server is no longer used.

The actual storage implementation
lives in app.storage.
"""

from app.storage import (
    ParquetStore,
    store,
)

__all__ = [
    "ParquetStore",
    "store",
]