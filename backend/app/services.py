MOCK_SENSORS = [
    {
        "id": "1001",
        "name": "Датчик дыма 1001",
        "type": "Датчик дыма",
        "objectName": "Коллектор №1",

        "currentState": "Норма",

        "riskScore": 0.82,
        "warning": True,
        "predictionStatus": "scored",

        "anomalyCandidate": True,

        "lastEventAt": "2026-09-22T15:40:00",

        "riskFactors": [
            "Рост количества тревог",
            "Частые изменения состояния"
        ]
    },

    {
        "id": "1002",
        "name": "Датчик температуры 1002",
        "type": "Датчик температуры",
        "objectName": "Коллектор №1",

        "currentState": "Норма",

        "riskScore": None,
        "warning": None,
        "predictionStatus": "not_available",

        "anomalyCandidate": True,

        "lastEventAt": "2026-09-22T15:32:00",

        "riskFactors": [
            "Обнаружено устойчивое изменение уровня"
        ]
    },

    {
        "id": "1003",
        "name": "Датчик дыма 1003",
        "type": "Датчик дыма",
        "objectName": "Коллектор №2",

        "currentState": "Неисправен",

        "riskScore": None,
        "warning": None,
        "predictionStatus": "already_faulty",

        "anomalyCandidate": False,

        "lastEventAt": "2026-09-22T15:20:00",

        "riskFactors": []
    },

    {
        "id": "1004",
        "name": "Контактный датчик 1004",
        "type": "Контактный датчик",
        "objectName": "Коллектор №3",

        "currentState": "Норма",

        "riskScore": 0.21,
        "warning": False,
        "predictionStatus": "scored",

        "anomalyCandidate": False,

        "lastEventAt": "2026-09-22T15:48:00",

        "riskFactors": []
    },

    {
        "id": "1005",
        "name": "Газовый датчик 1005",
        "type": "Газовый датчик",
        "objectName": "Коллектор №3",

        "currentState": "Норма",

        "riskScore": None,
        "warning": None,
        "predictionStatus": "stale_observation",

        "anomalyCandidate": False,

        "lastEventAt": "2026-09-20T11:10:00",

        "riskFactors": [
            "Давно не поступали новые данные"
        ]
    }
]


MOCK_HISTORY = {
    "1001": [
        {
            "timestamp": "2026-09-22T12:00:00",
            "state": "Норма",
            "alarm": False,
            "value": "Дыма нет",
            "unit": None
        },
        {
            "timestamp": "2026-09-22T13:00:00",
            "state": "Норма",
            "alarm": False,
            "value": "Дыма нет",
            "unit": None
        },
        {
            "timestamp": "2026-09-22T14:00:00",
            "state": "Тревога",
            "alarm": True,
            "value": "Обнаружен дым",
            "unit": None
        },
        {
            "timestamp": "2026-09-22T15:40:00",
            "state": "Норма",
            "alarm": False,
            "value": "Дыма нет",
            "unit": None
        }
    ],

    "1002": [
        {
            "timestamp": "2026-09-22T12:00:00",
            "state": "Норма",
            "alarm": False,
            "value": 18.4,
            "unit": "°C"
        },
        {
            "timestamp": "2026-09-22T13:00:00",
            "state": "Норма",
            "alarm": False,
            "value": 19.1,
            "unit": "°C"
        },
        {
            "timestamp": "2026-09-22T14:00:00",
            "state": "Норма",
            "alarm": False,
            "value": 22.8,
            "unit": "°C"
        },
        {
            "timestamp": "2026-09-22T15:32:00",
            "state": "Норма",
            "alarm": False,
            "value": 25.3,
            "unit": "°C"
        }
    ]
}


def get_sensors(
    group: str | None = None,
    limit: int = 100
):
    sensors = MOCK_SENSORS.copy()

    if group == "failed":
        sensors = [
            sensor
            for sensor in sensors
            if sensor["currentState"] == "Неисправен"
        ]

    elif group == "warning":
        sensors = [
            sensor
            for sensor in sensors
            if sensor["warning"] is True
        ]

    elif group == "anomaly":
        sensors = [
            sensor
            for sensor in sensors
            if sensor["anomalyCandidate"] is True
        ]

    return sensors[:limit]


def search_sensors(
    query: str,
    limit: int = 50
):
    query = query.casefold()

    result = []

    for sensor in MOCK_SENSORS:

        searchable_values = [
            sensor["id"],
            sensor["name"],
            sensor["type"],
            sensor["objectName"] or ""
        ]

        searchable_text = " ".join(
            searchable_values
        ).casefold()

        if query in searchable_text:
            result.append(sensor)

    return result[:limit]


def get_sensor(sensor_id: str):

    for sensor in MOCK_SENSORS:

        if sensor["id"] == sensor_id:
            return sensor

    return None


def get_sensor_history(sensor_id: str):

    return MOCK_HISTORY.get(
        sensor_id,
        []
    )


def get_sensor_assessment(sensor_id: str):

    sensor = get_sensor(sensor_id)

    if sensor is None:
        return None

    return {
        "id": sensor["id"],

        "currentState": sensor["currentState"],

        "riskScore": sensor["riskScore"],

        "warning": sensor["warning"],

        "predictionStatus": sensor[
            "predictionStatus"
        ],

        "anomalyCandidate": sensor[
            "anomalyCandidate"
        ],

        "riskFactors": sensor[
            "riskFactors"
        ]
    }


def get_dashboard_summary():

    total_sensors = len(
        MOCK_SENSORS
    )

    registered_faults = sum(
        sensor["currentState"] == "Неисправен"
        for sensor in MOCK_SENSORS
    )

    warnings = sum(
        sensor["warning"] is True
        for sensor in MOCK_SENSORS
    )

    anomaly_candidates = sum(
        sensor["anomalyCandidate"] is True
        for sensor in MOCK_SENSORS
    )

    prediction_unavailable = sum(
        sensor["predictionStatus"] != "scored"
        for sensor in MOCK_SENSORS
    )

    return {
        "totalSensors": total_sensors,

        "registeredFaults": registered_faults,

        "warnings": warnings,

        "anomalyCandidates": anomaly_candidates,

        "predictionUnavailable":
            prediction_unavailable
    }