from __future__ import annotations

from collections.abc import (
    Iterable,
    Mapping,
)
from typing import Any

import pandas as pd

from app.storage import store


def _record(
    value: Any,
) -> dict[str, Any]:

    if isinstance(
        value,
        Mapping,
    ):
        return dict(
            value
        )

    method = getattr(
        value,
        "to_record",
        None,
    )

    if callable(method):

        result = method()

        if not isinstance(
            result,
            dict,
        ):
            raise TypeError(
                "to_record() "
                "must return dict"
            )

        return result

    raise TypeError(
        "Unsupported ML "
        "record type: "
        f"{type(value)!r}"
    )


def _frame(
    records: Iterable[Any],
) -> pd.DataFrame:

    return pd.DataFrame(
        [
            _record(item)
            for item
            in records
        ]
    )


def save_normalized_events(
    events: Iterable[Any],
) -> int:

    return store.append_events(
        _frame(
            events
        )
    )


def save_episodes(
    episodes: Iterable[Any],
) -> int:

    return store.append_episodes(
        _frame(
            episodes
        )
    )


def save_forecast_decisions(
    decisions: Iterable[
        Mapping[str, Any]
    ],
) -> int:

    return store.append_forecasts(
        _frame(
            decisions
        )
    )