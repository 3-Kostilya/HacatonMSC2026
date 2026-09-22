"""Reuse existing Stage-1 parsing semantics while retaining rejected raw records."""

from stage1.normalization import EVENT_FIELDS, parse_alarm, parse_local_timestamp, parse_numeric
from stage1.ingestion.schemas import RAW_COLUMNS


def normalize_row(row: dict, row_id: int) -> dict:
    raw = [row[field] for field in EVENT_FIELDS]
    header = all(value.strip() == field for value, field in zip(raw, EVENT_FIELDS))
    flags = []
    try:
        timestamp = parse_local_timestamp(raw[2], raw[3])
    except ValueError as error:
        timestamp = None
        flags.append(str(error))
    try:
        alarm = parse_alarm(raw[4])
    except ValueError:
        alarm = None
        flags.append("invalid_alarm")
    number, number_flag = parse_numeric(raw[5])
    if number_flag:
        flags.append(number_flag)
    event_id, channel_id = raw[0].strip(), raw[1].strip()
    if not event_id:
        flags.append("missing_event_id")
    if not channel_id:
        flags.append("empty_channel_id")
    if not raw[5].strip():
        flags.append("empty_value")
    return {
        "row_id": row_id,
        "source": row["__source__"],
        "source_row": row["__source_row__"],
        **dict(zip(RAW_COLUMNS, raw)),
        "event_id": event_id or None,
        "channel_id": channel_id or None,
        "timestamp": timestamp,
        "alarm": alarm,
        "value_numeric": number,
        "value_state": raw[5] if number is None else None,
        "is_numeric": number is not None,
        "quality_flags": flags,
        "invalid": timestamp is None or alarm is None or not channel_id or not raw[5].strip(),
        "repeated_header": header,
    }
