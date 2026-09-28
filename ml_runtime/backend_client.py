from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


def _clean(value: Any) -> Any:
    """Convert pandas/numpy values to strict JSON-compatible Python values."""
    if value is None:
        return None
    if value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def request_json(
    backend_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: Any = None,
    timeout: int = 60,
) -> Any:
    base = backend_url.rstrip("/")
    url = f"{base}{path}"
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(_clean(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Backend returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot connect to backend at {base}: {exc.reason}") from exc


def check_backend(backend_url: str) -> None:
    result = request_json(backend_url, "/api/health")
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise RuntimeError(f"Unexpected backend health response: {result!r}")


def post_records(
    backend_url: str,
    path: str,
    records: list[dict[str, Any]],
    *,
    batch_size: int = 200,
) -> int:
    if not records:
        return 0
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    stored = 0
    for batch in _chunks(records, batch_size):
        result = request_json(
            backend_url,
            path,
            method="POST",
            payload=batch,
        )
        if not isinstance(result, dict) or result.get("status") != "stored":
            raise RuntimeError(f"Unexpected backend response for {path}: {result!r}")
        stored += int(result.get("rows", 0))
    return stored
