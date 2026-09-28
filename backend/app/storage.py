from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from app.config import (
    EPISODES_DIR,
    EVENTS_DIR,
    FORECASTS_DIR,
    IMPORTS_FILE,
    R6_THRESHOLD,
    SENSORS_FILE,
)
from app.models import (
    EPISODE_COLUMNS,
    EVENT_COLUMNS,
    FORECAST_COLUMNS,
    IMPORT_COLUMNS,
    SENSOR_COLUMNS,
)


_JSON_COLUMNS = {
    "quality_flags",
    "evidence",
    "observation_quality",
    "metadata",
    "score_contributions",
}

_DATETIME_COLUMNS = {
    "timestamp",
    "prediction_time",
    "history_through",
    "admission_through",
    "start_at",
    "confirmed_at",
    "end_at",
    "updated_at",
    "imported_at",
    "finished_at",
}

_BOOL_COLUMNS = {
    "alarm",
    "is_numeric",
    "threshold_crossed",
    "shadow_warning",
    "automatic_action_taken",
}

_NUMERIC_COLUMNS = {
    "row_id",
    "source_row",
    "value_numeric",
    "rule_score",
    "threshold",
    "score",
    "registered_fault_text_count_24h",
    "registered_fault_text_count_168h",
    "completed_episode_count_168h",
    "technical_message_count_24h",
    "research_score",
    "rows_count",
}


def _now_naive() -> datetime:
    return datetime.now().replace(tzinfo=None)


def _json_dump(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Keep valid JSON in canonical form when possible.
        try:
            parsed = json.loads(text)
        except Exception:
            return json.dumps(value, ensure_ascii=False)
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _json_load(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _nullable_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"true", "1", "t", "yes", "y", "да"}:
        return True
    if text in {"false", "0", "f", "no", "n", "нет"}:
        return False
    return None


class ParquetStore:
    """Small operational store built on Parquet.

    Large append-only tables are directories of immutable Parquet parts, so new
    batches do not rewrite the full history. Small mutable tables (sensors and
    import log) are atomic snapshot files.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

    @staticmethod
    def _empty(columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=columns)

    def _normalize(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = frame.copy()

        for column in columns:
            if column not in df.columns:
                df[column] = None

        df = df[columns]

        for column in _JSON_COLUMNS.intersection(df.columns):
            df[column] = df[column].map(_json_dump)

        for column in _DATETIME_COLUMNS.intersection(df.columns):
            df[column] = pd.to_datetime(df[column], errors="coerce")
            try:
                df[column] = df[column].dt.tz_localize(None)
            except (TypeError, AttributeError):
                pass

        for column in _BOOL_COLUMNS.intersection(df.columns):
            df[column] = df[column].map(_nullable_bool).astype("boolean")

        for column in _NUMERIC_COLUMNS.intersection(df.columns):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        return df

    @staticmethod
    def _atomic_write(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        df.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, path)

    def _append_part(self, directory: Path, frame: pd.DataFrame, columns: list[str]) -> int:
        if frame.empty:
            return 0
        df = self._normalize(frame, columns)
        stamp = _now_naive().strftime("%Y%m%d_%H%M%S_%f")
        target = directory / f"part-{stamp}-{uuid4().hex}.parquet"
        self._atomic_write(df, target)
        return len(df)

    @staticmethod
    def _read_snapshot(path: Path, columns: list[str]) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=columns)
        try:
            frame = pd.read_parquet(path, engine="pyarrow")
        except Exception:
            return pd.DataFrame(columns=columns)
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        return frame[columns]

    @staticmethod
    def _read_dataset(
        directory: Path,
        columns: list[str],
        *,
        channel_id: str | None = None,
        channel_ids: list[str] | None = None,
        time_column: str | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
    ) -> pd.DataFrame:
        files = list(directory.glob("*.parquet"))
        if not files:
            return pd.DataFrame(columns=columns)

        # Parquet parts can be written by different application versions.
        # Unifying fragment schemas lets old parts coexist with newly added
        # optional columns (for example research_score) without losing them.
        schemas = [pq.read_schema(path) for path in files]
        unified_schema = pa.unify_schemas(schemas)
        dataset = ds.dataset(
            [str(path) for path in files],
            format="parquet",
            schema=unified_schema,
        )
        filter_expression = None
        if channel_id is not None and "channel_id" in dataset.schema.names:
            filter_expression = ds.field("channel_id") == channel_id
        elif channel_ids and "channel_id" in dataset.schema.names:
            filter_expression = ds.field("channel_id").isin(channel_ids)

        if time_column and time_column in dataset.schema.names:
            if start_at is not None:
                expr = ds.field(time_column) >= pa.scalar(start_at)
                filter_expression = expr if filter_expression is None else filter_expression & expr
            if end_at is not None:
                expr = ds.field(time_column) <= pa.scalar(end_at)
                filter_expression = expr if filter_expression is None else filter_expression & expr

        available = [c for c in columns if c in dataset.schema.names]
        table = dataset.to_table(columns=available, filter=filter_expression)
        frame = table.to_pandas()
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        return frame[columns]

    def sensors(self) -> pd.DataFrame:
        return self._read_snapshot(SENSORS_FILE, SENSOR_COLUMNS)

    def events(self, channel_id: str | None = None) -> pd.DataFrame:
        return self._read_dataset(EVENTS_DIR, EVENT_COLUMNS, channel_id=channel_id)

    def forecasts(self, channel_id: str | None = None) -> pd.DataFrame:
        return self._read_dataset(FORECASTS_DIR, FORECAST_COLUMNS, channel_id=channel_id)

    def events_for_channels(
        self,
        channel_ids: Iterable[str],
        *,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
    ) -> pd.DataFrame:
        ids = sorted({str(value) for value in channel_ids if str(value).strip()})
        if not ids:
            return self._empty(EVENT_COLUMNS)
        return self._read_dataset(
            EVENTS_DIR,
            EVENT_COLUMNS,
            channel_ids=ids,
            time_column="timestamp",
            start_at=start_at,
            end_at=end_at,
        )

    def latest_event_times(self, channel_ids: Iterable[str]) -> dict[str, datetime]:
        frame = self.events_for_channels(channel_ids)
        if frame.empty:
            return {}
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame[frame["timestamp"].notna() & frame["channel_id"].notna()]
        if frame.empty:
            return {}
        latest = frame.groupby(frame["channel_id"].astype(str), sort=False)["timestamp"].max()
        return {str(channel): stamp.to_pydatetime().replace(tzinfo=None) for channel, stamp in latest.items()}

    def episodes(self, channel_id: str | None = None) -> pd.DataFrame:
        return self._read_dataset(EPISODES_DIR, EPISODE_COLUMNS, channel_id=channel_id)

    def imports(self) -> pd.DataFrame:
        return self._read_snapshot(IMPORTS_FILE, IMPORT_COLUMNS)

    def upsert_sensors(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        with self._lock:
            incoming = frame.copy()
            rename = {
                "id": "channel_id",
                "type": "sensor_type",
                "objectName": "object_name",
            }
            incoming = incoming.rename(columns={k: v for k, v in rename.items() if k in incoming.columns})
            if "channel_id" not in incoming.columns:
                raise ValueError("sensors require channel_id")
            if "sensor_type" not in incoming.columns:
                incoming["sensor_type"] = "unknown"
            if "updated_at" not in incoming.columns:
                incoming["updated_at"] = _now_naive()
            incoming = self._normalize(incoming, SENSOR_COLUMNS)
            incoming["channel_id"] = incoming["channel_id"].astype("string")
            incoming = incoming[incoming["channel_id"].notna() & (incoming["channel_id"].str.len() > 0)]

            current = self.sensors()
            merged = pd.concat([current, incoming], ignore_index=True)
            merged = merged.sort_values("updated_at", na_position="first")
            merged = merged.drop_duplicates(subset=["channel_id"], keep="last")
            self._atomic_write(self._normalize(merged, SENSOR_COLUMNS), SENSORS_FILE)
            return len(incoming)

    def _refresh_sensors_from(self, frame: pd.DataFrame, time_column: str | None = None) -> None:
        if frame.empty or "channel_id" not in frame.columns:
            return
        candidates = frame.copy()
        if time_column and time_column in candidates.columns:
            candidates[time_column] = pd.to_datetime(candidates[time_column], errors="coerce")
            candidates = candidates.sort_values(time_column).drop_duplicates("channel_id", keep="last")
            candidates["updated_at"] = candidates[time_column]
        else:
            candidates = candidates.drop_duplicates("channel_id", keep="last")
            candidates["updated_at"] = _now_naive()

        sensor_name = candidates.get("sensor_name") if "sensor_name" in candidates else None
        explicit_name = candidates.get("name") if "name" in candidates else None
        name = explicit_name if explicit_name is not None else sensor_name
        sensor_frame = pd.DataFrame(
            {
                "channel_id": candidates.get("channel_id"),
                "name": name,
                "sensor_type": candidates.get("sensor_type") if "sensor_type" in candidates else "unknown",
                "engineering_system_type": candidates.get("engineering_system_type") if "engineering_system_type" in candidates else None,
                "engineering_system_tag": candidates.get("engineering_system_tag") if "engineering_system_tag" in candidates else None,
                "sensor_name": sensor_name,
                "object_id": candidates.get("object_id") if "object_id" in candidates else None,
                "hierarchy_level": candidates.get("hierarchy_level") if "hierarchy_level" in candidates else None,
                "parent_object_id": candidates.get("parent_object_id") if "parent_object_id" in candidates else None,
                "object_kind": candidates.get("object_kind") if "object_kind" in candidates else None,
                "object_name": candidates.get("object_name") if "object_name" in candidates else None,
                "unit": candidates.get("unit") if "unit" in candidates else None,
                "updated_at": candidates.get("updated_at"),
            }
        )
        self.upsert_sensors(sensor_frame)

    def append_events(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        with self._lock:
            df = frame.copy()

            # Older NormalizedEvent / earlier backend names -> current M1 names.
            aliases = {
                "raw_value": "value_raw",
                "numeric_value": "value_numeric",
                "state": "value_state",
                "type": "sensor_type",
                "objectName": "object_name",
            }
            df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

            # Some prepared datasets use value_state as the textual state and do
            # not carry value_raw separately. Keep a lossless-enough fallback for
            # the operational store; the canonical M1 output already has value_raw.
            if "value_raw" not in df.columns and "value_state" in df.columns:
                df["value_raw"] = df["value_state"]
            if "value_state" not in df.columns and "value_raw" in df.columns:
                df["value_state"] = df["value_raw"]
            if "is_numeric" not in df.columns:
                if "value_numeric" in df.columns:
                    numeric = pd.to_numeric(df["value_numeric"], errors="coerce")
                    df["is_numeric"] = numeric.notna()
                else:
                    df["is_numeric"] = False

            required = {"channel_id", "timestamp", "value_raw", "alarm"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"events missing columns: {', '.join(sorted(missing))}")

            if "source" not in df.columns:
                df["source"] = "backend_import"
            if "sensor_type" not in df.columns:
                df["sensor_type"] = "unknown"
            if "event_id" not in df.columns:
                df["event_id"] = None
            if "quality_flags" not in df.columns:
                df["quality_flags"] = [[] for _ in range(len(df))]

            normalized = self._normalize(df, EVENT_COLUMNS)
            normalized = normalized[normalized["channel_id"].notna() & normalized["timestamp"].notna()]
            count = self._append_part(EVENTS_DIR, normalized, EVENT_COLUMNS)
            self._refresh_sensors_from(df, "timestamp")
            return count

    def append_forecasts(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        with self._lock:
            df = frame.copy()
            required = {"channel_id", "prediction_time", "sensor_type", "prediction_status"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"forecasts missing columns: {', '.join(sorted(missing))}")
            if "threshold" not in df.columns:
                df["threshold"] = R6_THRESHOLD
            else:
                df["threshold"] = df["threshold"].fillna(R6_THRESHOLD)
            normalized = self._normalize(df, FORECAST_COLUMNS)
            normalized = normalized[normalized["channel_id"].notna() & normalized["prediction_time"].notna()]
            count = self._append_part(FORECASTS_DIR, normalized, FORECAST_COLUMNS)
            self._refresh_sensors_from(df, "prediction_time")
            return count

    def append_episodes(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        with self._lock:
            df = frame.copy()
            required = {
                "episode_id",
                "channel_id",
                "sensor_type",
                "sensor_group",
                "anomaly_type",
                "decision",
                "start_at",
                "confirmed_at",
                "ruleset_version",
            }
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"episodes missing columns: {', '.join(sorted(missing))}")
            normalized = self._normalize(df, EPISODE_COLUMNS)
            normalized = normalized[normalized["channel_id"].notna() & normalized["episode_id"].notna()]
            count = self._append_part(EPISODES_DIR, normalized, EPISODE_COLUMNS)
            self._refresh_sensors_from(df, "confirmed_at")
            return count

    def add_import_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            current = self.imports()
            row = pd.DataFrame([record])
            row = self._normalize(row, IMPORT_COLUMNS)
            merged = pd.concat([current, row], ignore_index=True)
            merged = merged.drop_duplicates(subset=["batch_id"], keep="last")
            self._atomic_write(self._normalize(merged, IMPORT_COLUMNS), IMPORTS_FILE)

    def update_import_record(self, batch_id: str, **updates: Any) -> None:
        with self._lock:
            current = self.imports()
            if current.empty:
                return
            mask = current["batch_id"].astype(str) == str(batch_id)
            if not mask.any():
                return
            for key, value in updates.items():
                if key in current.columns:
                    current.loc[mask, key] = value
            self._atomic_write(self._normalize(current, IMPORT_COLUMNS), IMPORTS_FILE)

    def has_file_hash(self, file_hash: str) -> bool:
        imports = self.imports()
        if imports.empty:
            return False
        return bool((imports["file_hash"].astype(str) == str(file_hash)).any())

    @staticmethod
    def decode_json(value: Any, default: Any) -> Any:
        return _json_load(value, default)


store = ParquetStore()
