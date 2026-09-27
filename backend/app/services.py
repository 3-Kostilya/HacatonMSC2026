from __future__ import annotations

from typing import Any

import pandas as pd

from app.config import R6_THRESHOLD
from app.storage import ParquetStore

_CONTRIBUTION_LABELS = {
    "registered_fault_text_count_24h":
        "Записи «Неисправен» за 24 ч",

    "registered_fault_text_count_168h":
        "Записи «Неисправен» за 7 дней",

    "completed_episode_count_168h":
        "Завершённые эпизоды за 7 дней",

    "technical_message_count_24h":
        "Технические сообщения за 24 ч",
}


def _clean(
    value: Any,
) -> Any:

    if value is None:
        return None

    if (
        isinstance(value, float)
        and pd.isna(value)
    ):
        return None

    if not isinstance(
        value,
        (
            list,
            dict,
            tuple,
        ),
    ):
        try:
            if pd.isna(value):
                return None

        except (
            TypeError,
            ValueError,
        ):
            pass

    return value


def _text(
    value: Any,
    default: str = "",
) -> str:

    value = _clean(value)

    if value is None:
        return default

    return str(value)


def _iso(
    value: Any,
) -> str | None:

    value = _clean(value)

    if value is None:
        return None

    stamp = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(stamp):
        return None

    return (
        stamp
        .to_pydatetime()
        .replace(
            tzinfo=None
        )
        .isoformat(
            timespec="seconds"
        )
    )


def _latest(
    frame: pd.DataFrame,
    time_column: str,
) -> pd.DataFrame:

    if frame.empty:
        return frame

    df = frame.copy()

    df[
        time_column
    ] = pd.to_datetime(
        df[
            time_column
        ],
        errors="coerce",
    )

    df = df[
        df[
            time_column
        ].notna()
    ]

    if df.empty:
        return df

    return (
        df
        .sort_values(
            time_column
        )
        .drop_duplicates(
            "channel_id",
            keep="last",
        )
    )


def _faulty_state(
    value: Any,
) -> bool:

    return (
        "неисправ"
        in _text(
            value
        ).casefold()
    )


def _frontend_prediction_status(
    forecast: (
        dict[str, Any]
        | None
    ),
) -> str:

    if not forecast:
        return "not_available"

    if (
        _text(
            forecast.get(
                "prediction_status"
            )
        ).casefold()
        == "scored"
    ):
        return "scored"

    reason = _text(
        forecast.get(
            "unavailable_reason"
        )
    ).casefold()

    if (
        "already" in reason
        or "fault" in reason
        or "неисправ" in reason
    ):
        return "already_faulty"

    if "unknown" in reason:
        return "unknown_state"

    if (
        "stale" in reason
        or "old" in reason
    ):
        return "stale_observation"

    if (
        "history" in reason
        or "insufficient" in reason
        or "coverage" in reason
    ):
        return "insufficient_history"

    return "not_available"


def _compat_risk_index(
    forecast: (
        dict[str, Any]
        | None
    ),
) -> float | None:

    """
    Старый frontend ожидает riskScore 0..1.

    Текущий ML rule_score НЕ является
    вероятностью физической поломки.

    Поэтому временно используется
    индекс относительно порога:

        rule_score / threshold
    """

    if not forecast:
        return None

    score = _clean(
        forecast.get(
            "rule_score"
        )
    )

    threshold = (
        _clean(
            forecast.get(
                "threshold"
            )
        )
        or R6_THRESHOLD
    )

    if (
        score is None
        or not threshold
    ):
        return None

    try:
        return max(
            0.0,
            min(
                float(score)
                / float(threshold),
                1.0,
            ),
        )

    except (
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        return None


def _risk_factors(
    store: ParquetStore,
    forecast: (
        dict[str, Any]
        | None
    ),
    episode: (
        dict[str, Any]
        | None
    ),
) -> list[str]:

    factors: list[str] = []

    if forecast:

        contributions = (
            store.decode_json(
                forecast.get(
                    "score_contributions"
                ),
                {},
            )
        )

        if isinstance(
            contributions,
            dict,
        ):

            for (
                key,
                value,
            ) in contributions.items():

                try:
                    numeric = float(
                        value
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if numeric == 0:
                    continue

                label = (
                    _CONTRIBUTION_LABELS
                    .get(
                        key,
                        key,
                    )
                )

                factors.append(
                    f"{label}: "
                    f"вклад {numeric:g}"
                )

        warning_reason = _clean(
            forecast.get(
                "warning_reason"
            )
        )

        if (
            warning_reason
            and warning_reason
            not in {
                "recorded_shadow_warning",
                "below_frozen_threshold",
                "no_prediction",
            }
        ):
            factors.append(
                "ML: "
                f"{warning_reason}"
            )

        unavailable = _clean(
            forecast.get(
                "unavailable_reason"
            )
        )

        if unavailable:
            factors.append(
                "Прогноз недоступен: "
                f"{unavailable}"
            )

    if (
        episode
        and _text(
            episode.get(
                "decision"
            )
        )
        == "candidate"
    ):

        evidence = (
            store.decode_json(
                episode.get(
                    "evidence"
                ),
                [],
            )
        )

        if isinstance(
            evidence,
            list,
        ):
            factors.extend(
                str(item)
                for item in evidence
                if str(item).strip()
            )

    return list(
        dict.fromkeys(
            factors
        )
    )[:10]


def build_sensor_view(
    store: ParquetStore,
) -> list[dict[str, Any]]:

    sensors = store.sensors()

    events = store.events()

    forecasts = (
        store.forecasts()
    )

    episodes = (
        store.episodes()
    )

    latest_events = _latest(
        events,
        "timestamp",
    )

    latest_forecasts = _latest(
        forecasts,
        "prediction_time",
    )

    latest_episodes = _latest(
        episodes,
        "confirmed_at",
    )

    ids: set[str] = set()

    for frame in (
        sensors,
        latest_events,
        latest_forecasts,
        latest_episodes,
    ):

        if (
            not frame.empty
            and "channel_id"
            in frame.columns
        ):

            ids.update(
                str(value)
                for value
                in frame[
                    "channel_id"
                ]
                .dropna()
                .astype(str)
            )

    sensor_map = {
        str(
            row[
                "channel_id"
            ]
        ):
            row.to_dict()

        for _, row
        in sensors.iterrows()
    }

    event_map = {
        str(
            row[
                "channel_id"
            ]
        ):
            row.to_dict()

        for _, row
        in latest_events.iterrows()
    }

    forecast_map = {
        str(
            row[
                "channel_id"
            ]
        ):
            row.to_dict()

        for _, row
        in latest_forecasts.iterrows()
    }

    episode_map = {
        str(
            row[
                "channel_id"
            ]
        ):
            row.to_dict()

        for _, row
        in latest_episodes.iterrows()
    }

    result: list[
        dict[str, Any]
    ] = []

    for channel_id in sorted(
        ids
    ):

        sensor = sensor_map.get(
            channel_id,
            {},
        )

        event = event_map.get(
            channel_id
        )

        forecast = forecast_map.get(
            channel_id
        )

        episode = episode_map.get(
            channel_id
        )

        sensor_type = (
            _text(
                sensor.get(
                    "sensor_type"
                )
            )

            or _text(
                event.get(
                    "sensor_type"
                )
                if event
                else None
            )

            or _text(
                forecast.get(
                    "sensor_type"
                )
                if forecast
                else None
            )

            or _text(
                episode.get(
                    "sensor_type"
                )
                if episode
                else None
            )

            or "Неизвестный тип"
        )

        if event:

            state_value = _clean(
                event.get(
                    "value_state"
                )
            )

            raw_value = _clean(
                event.get(
                    "value_raw"
                )
            )

            current_state = _text(
                (
                    state_value
                    if state_value
                    is not None
                    else raw_value
                ),
                "Нет данных",
            )

        else:
            current_state = (
                "Нет данных"
            )

        object_name = _clean(
            sensor.get(
                "object_name"
            )
        )

        name = (
            _text(
                sensor.get(
                    "name"
                )
            )

            or _text(
                sensor.get(
                    "sensor_name"
                )
            )

            or (
                f"{sensor_type} · "
                f"{channel_id}"
            )
        )

        threshold = _clean(
            forecast.get(
                "threshold"
            )
            if forecast
            else None
        )

        threshold = (
            float(threshold)
            if threshold
            is not None
            else (
                R6_THRESHOLD
                if forecast
                else None
            )
        )

        rule_score = _clean(
            forecast.get(
                "rule_score"
            )
            if forecast
            else None
        )

        rule_score = (
            float(rule_score)
            if rule_score
            is not None
            else None
        )

        threshold_crossed = (
            _clean(
                forecast.get(
                    "threshold_crossed"
                )
                if forecast
                else None
            )
        )

        warning = _clean(
            forecast.get(
                "shadow_warning"
            )
            if forecast
            else None
        )

        anomaly_candidate = bool(
            episode
            and _text(
                episode.get(
                    "decision"
                )
            )
            == "candidate"
        )

        result.append(
            {
                "id":
                    channel_id,

                "name":
                    name,

                "type":
                    sensor_type,

                "objectName":
                    (
                        str(
                            object_name
                        )
                        if object_name
                        is not None
                        else None
                    ),

                "currentState":
                    current_state,

                "riskScore":
                    _compat_risk_index(
                        forecast
                    ),

                "warning":
                    (
                        bool(warning)
                        if warning
                        is not None
                        else None
                    ),

                "predictionStatus":
                    _frontend_prediction_status(
                        forecast
                    ),

                "anomalyCandidate":
                    anomaly_candidate,

                "ruleScore":
                    rule_score,

                "threshold":
                    threshold,

                "thresholdCrossed":
                    (
                        bool(
                            threshold_crossed
                        )
                        if threshold_crossed
                        is not None
                        else None
                    ),

                "mlPredictionStatus":
                    (
                        _text(
                            forecast.get(
                                "prediction_status"
                            )
                        )
                        if forecast
                        else None
                    ),

                "lastEventAt":
                    (
                        _iso(
                            event.get(
                                "timestamp"
                            )
                        )
                        if event
                        else None
                    ),

                "riskFactors":
                    _risk_factors(
                        store,
                        forecast,
                        episode,
                    ),

                "objectId":
                    (
                        _text(
                            sensor.get(
                                "object_id"
                            )
                        )

                        or _text(
                            event.get(
                                "object_id"
                            )
                            if event
                            else None
                        )

                        or None
                    ),

                "sensorType":
                    sensor_type,

                "_faulty":
                    _faulty_state(
                        current_state
                    ),

                "_forecast":
                    forecast,

                "_episode":
                    episode,
            }
        )

    return result


def dashboard_summary(
    store: ParquetStore,
) -> dict[str, int]:

    items = build_sensor_view(
        store
    )

    return {
        "totalSensors":
            len(items),

        "registeredFaults":
            sum(
                1
                for item in items
                if item[
                    "_faulty"
                ]
            ),

        "warnings":
            sum(
                1
                for item in items
                if item[
                    "warning"
                ] is True
            ),

        "anomalyCandidates":
            sum(
                1
                for item in items
                if item[
                    "anomalyCandidate"
                ]
            ),

        "predictionUnavailable":
            sum(
                1
                for item in items
                if item[
                    "predictionStatus"
                ]
                != "scored"
            ),
    }


def sensor_list(
    store: ParquetStore,
    group: str | None = None,
) -> list[dict[str, Any]]:

    items = build_sensor_view(
        store
    )

    if group == "failed":
        items = [
            item
            for item in items
            if item["_faulty"]
        ]

    elif group == "warning":
        items = [
            item
            for item in items
            if item[
                "warning"
            ] is True
        ]

    elif group == "anomaly":
        items = [
            item
            for item in items
            if item[
                "anomalyCandidate"
            ]
        ]

    def sort_key(
        item: dict[str, Any],
    ) -> tuple[
        int,
        int,
        float,
        str,
    ]:

        return (
            (
                1
                if item[
                    "_faulty"
                ]
                else 0
            ),

            (
                1
                if item[
                    "warning"
                ]
                else 0
            ),

            float(
                item[
                    "ruleScore"
                ]
                or -1
            ),

            item["id"],
        )

    items.sort(
        key=sort_key,
        reverse=True,
    )

    for item in items:

        item.pop(
            "_faulty",
            None,
        )

        item.pop(
            "_forecast",
            None,
        )

        item.pop(
            "_episode",
            None,
        )

        item.pop(
            "lastEventAt",
            None,
        )

        item.pop(
            "riskFactors",
            None,
        )

        item.pop(
            "objectId",
            None,
        )

        item.pop(
            "sensorType",
            None,
        )

    return items


def search_sensors(
    store: ParquetStore,
    query: str,
) -> list[dict[str, Any]]:

    q = (
        query
        .strip()
        .casefold()
    )

    if not q:
        return sensor_list(
            store
        )

    items = sensor_list(
        store
    )

    return [
        item
        for item in items
        if (
            q
            in item[
                "id"
            ].casefold()

            or q
            in item[
                "name"
            ].casefold()

            or q
            in item[
                "type"
            ].casefold()

            or q
            in (
                item.get(
                    "objectName"
                )
                or ""
            ).casefold()
        )
    ]


def sensor_details(
    store: ParquetStore,
    sensor_id: str,
) -> dict[str, Any] | None:

    item = next(
        (
            item
            for item
            in build_sensor_view(
                store
            )
            if item[
                "id"
            ]
            == sensor_id
        ),
        None,
    )

    if item is None:
        return None

    item.pop(
        "_faulty",
        None,
    )

    item.pop(
        "_forecast",
        None,
    )

    item.pop(
        "_episode",
        None,
    )

    return item


def sensor_history(
    store: ParquetStore,
    sensor_id: str,
    limit: int = 100,
) -> list[dict[str, Any]]:

    frame = store.events(
        sensor_id
    )

    if frame.empty:
        return []

    frame[
        "timestamp"
    ] = pd.to_datetime(
        frame[
            "timestamp"
        ],
        errors="coerce",
    )

    frame = (
        frame[
            frame[
                "timestamp"
            ].notna()
        ]
        .sort_values(
            "timestamp",
            ascending=False,
        )
        .head(limit)
    )

    result = []

    for _, row in (
        frame.iterrows()
    ):

        raw = _clean(
            row.get(
                "value_raw"
            )
        )

        state = _clean(
            row.get(
                "value_state"
            )
        )

        numeric = _clean(
            row.get(
                "value_numeric"
            )
        )

        state_text = (
            state
            if state
            is not None
            else raw
        )

        result.append(
            {
                "timestamp":
                    (
                        _iso(
                            row.get(
                                "timestamp"
                            )
                        )
                        or ""
                    ),

                "state":
                    _text(
                        state_text,
                        "Нет данных",
                    ),

                "alarm":
                    bool(
                        _clean(
                            row.get(
                                "alarm"
                            )
                        )
                        or False
                    ),

                "value":
                    (
                        numeric
                        if numeric
                        is not None
                        else state_text
                    ),

                "unit":
                    (
                        _text(
                            row.get(
                                "unit"
                            )
                        )
                        or None
                    ),
            }
        )

    return result


def sensor_assessment(
    store: ParquetStore,
    sensor_id: str,
) -> dict[str, Any] | None:

    item = next(
        (
            item
            for item
            in build_sensor_view(
                store
            )
            if item[
                "id"
            ]
            == sensor_id
        ),
        None,
    )

    if item is None:
        return None

    forecast = item.pop(
        "_forecast",
        None,
    )

    item.pop(
        "_episode",
        None,
    )

    item.pop(
        "_faulty",
        None,
    )

    contributions = None

    if forecast:

        parsed = (
            store.decode_json(
                forecast.get(
                    "score_contributions"
                ),
                None,
            )
        )

        if isinstance(
            parsed,
            dict,
        ):

            contributions = {}

            for (
                key,
                value,
            ) in parsed.items():

                try:
                    contributions[
                        str(key)
                    ] = float(
                        value
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

    return {
        "id":
            item["id"],

        "currentState":
            item[
                "currentState"
            ],

        "riskScore":
            item[
                "riskScore"
            ],

        "warning":
            item[
                "warning"
            ],

        "predictionStatus":
            item[
                "predictionStatus"
            ],

        "anomalyCandidate":
            item[
                "anomalyCandidate"
            ],

        "riskFactors":
            item[
                "riskFactors"
            ],

        "ruleScore":
            item[
                "ruleScore"
            ],

        "threshold":
            item[
                "threshold"
            ],

        "thresholdCrossed":
            item[
                "thresholdCrossed"
            ],

        "mlPredictionStatus":
            item[
                "mlPredictionStatus"
            ],

        "predictionTime":
            (
                _iso(
                    forecast.get(
                        "prediction_time"
                    )
                )
                if forecast
                else None
            ),

        "admissionStatus":
            (
                _text(
                    forecast.get(
                        "admission_status"
                    )
                )
                if forecast
                else None
            ),

        "admissionReason":
            (
                (
                    _text(
                        forecast.get(
                            "admission_reason"
                        )
                    )
                    or None
                )
                if forecast
                else None
            ),

        "unavailableReason":
            (
                (
                    _text(
                        forecast.get(
                            "unavailable_reason"
                        )
                    )
                    or None
                )
                if forecast
                else None
            ),

        "warningReason":
            (
                (
                    _text(
                        forecast.get(
                            "warning_reason"
                        )
                    )
                    or None
                )
                if forecast
                else None
            ),

        "policyVersion":
            (
                (
                    _text(
                        forecast.get(
                            "policy_version"
                        )
                    )
                    or None
                )
                if forecast
                else None
            ),

        "scoreContributions":
            contributions,
    }