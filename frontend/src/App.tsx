import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";

import {
  getDashboardSummary,
  getHealth,
  getSensorAssessment,
  getSensorDetails,
  getSensorHistory,
  getSensors,
  searchSensors,
} from "./api/api";

import type {
  DashboardSummary,
  SensorAssessment,
  SensorDetails,
  SensorEvent,
  SensorGroup,
  SensorListItem,
} from "./types/api";

import "./App.css";


type Filter =
  | "all"
  | SensorGroup;

type Theme =
  | "dark"
  | "light";


/* =========================================================
   ICONS
========================================================= */


function SunIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
    >
      <circle
        cx="12"
        cy="12"
        r="3.5"
      />

      <path
        d="
          M12 2.5V5
          M12 19V21.5
          M4.5 4.5L6.3 6.3
          M17.7 17.7L19.5 19.5
          M2.5 12H5
          M19 12H21.5
          M4.5 19.5L6.3 17.7
          M17.7 6.3L19.5 4.5
        "
      />
    </svg>
  );
}


function MoonIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
    >
      <path
        d="
          M20.2 15.2
          A8.6 8.6 0 0 1
          8.8 3.8
          A8.6 8.6 0 1 0
          20.2 15.2
          Z
        "
      />
    </svg>
  );
}


function CloseIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
    >
      <path d="M7 7L17 17" />
      <path d="M17 7L7 17" />
    </svg>
  );
}


function RefreshIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
    >
      <path
        d="
          M20 6V11H15
        "
      />

      <path
        d="
          M18.2 8
          A7.5 7.5 0 1 0
          19 15
        "
      />
    </svg>
  );
}


/* =========================================================
   HELPERS
========================================================= */

function formatDate(
  value: string | null
) {
  if (!value) {
    return "Нет данных";
  }

  const date =
    new Date(value);

  if (
    Number.isNaN(
      date.getTime()
    )
  ) {
    return value;
  }

  return new Intl.DateTimeFormat(
    "ru-RU",
    {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }
  ).format(date);
}


function predictionStatusText(
  status: string
) {
  const names:
    Record<string, string> = {

    scored:
      "Оценка рассчитана",

    already_faulty:
      "Уже неисправен",

    unknown_state:
      "Неизвестное состояние",

    stale_observation:
      "Данные устарели",

    insufficient_history:
      "Недостаточно истории",

    not_available:
      "Прогноз недоступен",
  };

  return (
    names[status] ??
    status
  );
}


function riskText(
  score: number | null
) {
  if (score === null) {
    return "Нет оценки";
  }

  return `${Math.round(
    score * 100
  )}/100`;
}


function getInitialTheme(): Theme {
  const saved =
    localStorage.getItem(
      "theme"
    );

  if (
    saved === "light" ||
    saved === "dark"
  ) {
    return saved;
  }

  return "dark";
}


/* =========================================================
   APP
========================================================= */

function App() {

  /* Theme */

  const [
    theme,
    setTheme,
  ] =
    useState<Theme>(
      getInitialTheme
    );


  /* Connection */

  const [
    backendOnline,
    setBackendOnline,
  ] =
    useState(false);


  /* Dashboard */

  const [
    summary,
    setSummary,
  ] =
    useState<DashboardSummary | null>(
      null
    );


  /* Sensors */

  const [
    sensors,
    setSensors,
  ] =
    useState<SensorListItem[]>(
      []
    );

  const [
    selectedSensor,
    setSelectedSensor,
  ] =
    useState<SensorDetails | null>(
      null
    );

  const [
    activeSensorId,
    setActiveSensorId,
  ] =
    useState<string | null>(
      null
    );

  const [
    assessment,
    setAssessment,
  ] =
    useState<SensorAssessment | null>(
      null
    );

  const [
    history,
    setHistory,
  ] =
    useState<SensorEvent[]>(
      []
    );


  /* Search / filter */

  const [
    filter,
    setFilter,
  ] =
    useState<Filter>(
      "all"
    );

  const [
    search,
    setSearch,
  ] =
    useState("");


  /* UI */

  const [
    initialLoading,
    setInitialLoading,
  ] =
    useState(true);

  const [
    detailsLoading,
    setDetailsLoading,
  ] =
    useState(false);

  const [
    refreshing,
    setRefreshing,
  ] =
    useState(false);

  const [
    error,
    setError,
  ] =
    useState<string | null>(
      null
    );


  /*
   * Номер последнего запроса датчика.
   * Не позволяет старому запросу
   * перезаписать более новый.
   */
  const sensorRequestId =
    useRef(0);


  /* =======================================================
     THEME
  ======================================================= */

  function toggleTheme() {

    const nextTheme: Theme =
      theme === "dark"
        ? "light"
        : "dark";

    setTheme(
      nextTheme
    );
  }


  useEffect(() => {

    document.documentElement.dataset.theme =
      theme;

    localStorage.setItem(
      "theme",
      theme
    );

    const metaTheme =
      document.querySelector(
        'meta[name="theme-color"]'
      );

    metaTheme?.setAttribute(
      "content",
      theme === "dark"
        ? "#15191e"
        : "#dde2e6"
    );

  }, [theme]);


  /* =======================================================
     OVERVIEW
  ======================================================= */

  async function loadOverview(
    selectedFilter: Filter
  ) {

    const group =
      selectedFilter === "all"
        ? undefined
        : selectedFilter;

    const [
      dashboardData,
      sensorData,
    ] =
      await Promise.all([
        getDashboardSummary(),
        getSensors(group),
      ]);

    setSummary(
      dashboardData
    );

    setSensors(
      sensorData
    );
  }


  /* =======================================================
     INITIAL LOAD
  ======================================================= */

  useEffect(() => {

    let cancelled =
      false;


    async function initialize() {

      try {

        setInitialLoading(
          true
        );

        setError(
          null
        );

        await getHealth();

        if (cancelled) {
          return;
        }

        setBackendOnline(
          true
        );


        const [
          dashboardData,
          sensorData,
        ] =
          await Promise.all([
            getDashboardSummary(),
            getSensors(),
          ]);


        if (cancelled) {
          return;
        }

        setSummary(
          dashboardData
        );

        setSensors(
          sensorData
        );

      } catch (err) {

        if (cancelled) {
          return;
        }

        setBackendOnline(
          false
        );

        setError(
          err instanceof Error
            ? err.message
            : "Не удалось подключиться к backend"
        );

      } finally {

        if (!cancelled) {

          setInitialLoading(
            false
          );
        }
      }
    }


    void initialize();


    return () => {

      cancelled =
        true;
    };

  }, []);


  /* =======================================================
     SENSOR
  ======================================================= */

  async function selectSensor(
    sensorId: string
  ) {

    const requestId =
      ++sensorRequestId.current;


    /*
     * Выделение карточки происходит сразу,
     * но старая правая панель не исчезает.
     */
    setActiveSensorId(
      sensorId
    );

    setDetailsLoading(
      true
    );

    setError(
      null
    );


    try {

      const [
        sensorData,
        historyData,
        assessmentData,
      ] =
        await Promise.all([
          getSensorDetails(
            sensorId
          ),

          getSensorHistory(
            sensorId
          ),

          getSensorAssessment(
            sensorId
          ),
        ]);


      /*
       * Если пользователь уже успел
       * выбрать другой датчик,
       * старый ответ игнорируем.
       */
      if (
        requestId !==
        sensorRequestId.current
      ) {
        return;
      }


      setSelectedSensor(
        sensorData
      );

      setHistory(
        historyData
      );

      setAssessment(
        assessmentData
      );

    } catch (err) {

      if (
        requestId !==
        sensorRequestId.current
      ) {
        return;
      }

      setError(
        err instanceof Error
          ? err.message
          : "Не удалось загрузить данные датчика"
      );

    } finally {

      if (
        requestId ===
        sensorRequestId.current
      ) {

        setDetailsLoading(
          false
        );
      }
    }
  }


  /* =======================================================
     FILTER
  ======================================================= */

  async function changeFilter(
    newFilter: Filter
  ) {

    try {

      setError(
        null
      );

      setFilter(
        newFilter
      );

      setSearch(
        ""
      );

      const group =
        newFilter === "all"
          ? undefined
          : newFilter;

      const data =
        await getSensors(
          group
        );

      setSensors(
        data
      );

    } catch (err) {

      setError(
        err instanceof Error
          ? err.message
          : "Не удалось загрузить датчики"
      );
    }
  }


  /* =======================================================
     SEARCH
  ======================================================= */

  async function handleSearch(
    event: FormEvent
  ) {

    event.preventDefault();

    const query =
      search.trim();


    if (!query) {

      try {

        setError(
          null
        );

        await loadOverview(
          filter
        );

      } catch (err) {

        setError(
          err instanceof Error
            ? err.message
            : "Не удалось загрузить датчики"
        );
      }

      return;
    }


    try {

      setError(
        null
      );

      const data =
        await searchSensors(
          query
        );

      setSensors(
        data
      );

    } catch (err) {

      setError(
        err instanceof Error
          ? err.message
          : "Ошибка поиска"
      );
    }
  }


  /* =======================================================
     REFRESH
  ======================================================= */

  async function refresh() {

    if (refreshing) {
      return;
    }


    try {

      setRefreshing(
        true
      );

      setError(
        null
      );


      await getHealth();

      setBackendOnline(
        true
      );


      if (
        search.trim()
      ) {

        const [
          sensorData,
          dashboardData,
        ] =
          await Promise.all([
            searchSensors(
              search.trim()
            ),

            getDashboardSummary(),
          ]);

        setSensors(
          sensorData
        );

        setSummary(
          dashboardData
        );

      } else {

        await loadOverview(
          filter
        );
      }


      if (
        selectedSensor
      ) {

        await selectSensor(
          selectedSensor.id
        );
      }

    } catch (err) {

      setBackendOnline(
        false
      );

      setError(
        err instanceof Error
          ? err.message
          : "Не удалось обновить данные"
      );

    } finally {

      setRefreshing(
        false
      );
    }
  }


  /* =======================================================
     CURRENT DATA
  ======================================================= */

  const currentRiskScore =
    assessment?.riskScore ??
    selectedSensor?.riskScore ??
    null;


  const currentRiskFactors =
    assessment?.riskFactors ??
    selectedSensor?.riskFactors ??
    [];


  const currentState =
    assessment?.currentState ??
    selectedSensor?.currentState ??
    "";


  /* =======================================================
     RENDER
  ======================================================= */

  return (

    <div className="app">


      {/* HEADER */}

      <header className="header">

        <div className="brand">

          <div className="brand-mark">
            <img
              src="/logo.png"
              alt="Логотип системы"
              className="brand-logo"
            />

          </div>


          <div className="brand-text">

            <h1>
              Мониторинг инженерной инфраструктуры
            </h1>

            <p>
              Состояние, аномалии и прогноз работы датчиков
            </p>

          </div>

        </div>


        <div className="header-actions">

          <div
            className={
              backendOnline
                ? "connection online"
                : "connection offline"
            }
          >

            <span className="connection-dot" />

            <span>
              {backendOnline
                ? "Backend подключён"
                : "Нет соединения"}
            </span>

          </div>


          <button
            className="icon-button theme-button"
            onClick={toggleTheme}
            aria-label={
              theme === "dark"
                ? "Включить светлую тему"
                : "Включить тёмную тему"
            }
            title={
              theme === "dark"
                ? "Светлая тема"
                : "Тёмная тема"
            }
          >

            {theme === "dark"
              ? <SunIcon />
              : <MoonIcon />}

          </button>


          <button
            className={
              refreshing
                ? "secondary-button refreshing"
                : "secondary-button"
            }
            onClick={() => {
              void refresh();
            }}
            disabled={
              refreshing
            }
          >

            <span className="refresh-icon">
              <RefreshIcon />
            </span>

            <span>
              Обновить
            </span>

          </button>

        </div>

      </header>


      {/* ERROR */}

      {error && (

        <div className="error-banner">

          <span>
            {error}
          </span>


          <button
            className="error-close"
            onClick={() => {
              setError(null);
            }}
            aria-label="Закрыть сообщение"
          >

            <CloseIcon />

          </button>

        </div>
      )}


      {/* SUMMARY */}

      <section className="summary-grid">

        <article className="summary-card">

          <span>
            Всего датчиков
          </span>

          <strong>
            {summary?.totalSensors ?? "—"}
          </strong>

        </article>


        <article className="summary-card">

          <span>
            Неисправны
          </span>

          <strong>
            {summary?.registeredFaults ?? "—"}
          </strong>

        </article>


        <article className="summary-card">

          <span>
            Предупреждения
          </span>

          <strong>
            {summary?.warnings ?? "—"}
          </strong>

        </article>


        <article className="summary-card">

          <span>
            Аномалии
          </span>

          <strong>
            {summary?.anomalyCandidates ?? "—"}
          </strong>

        </article>


        <article className="summary-card">

          <span>
            Без прогноза
          </span>

          <strong>
            {summary?.predictionUnavailable ?? "—"}
          </strong>

        </article>

      </section>


      {/* WORKSPACE */}

      <main className="workspace">


        {/* LEFT */}

        <section className="sensor-panel">

          <div className="panel-header">

            <div>

              <h2>
                Датчики
              </h2>

              <p>
                Найдено: {sensors.length}
              </p>

            </div>

          </div>


          <form
            className="search"
            onSubmit={(event) => {
              void handleSearch(
                event
              );
            }}
          >

            <input
              value={search}
              onChange={(event) => {
                setSearch(
                  event.target.value
                );
              }}
              placeholder="ID, название, тип или объект"
            />

            <button type="submit">
              Найти
            </button>

          </form>


          <div className="filters">

            <button
              className={
                filter === "all"
                  ? "filter active"
                  : "filter"
              }
              onClick={() => {
                void changeFilter(
                  "all"
                );
              }}
            >
              Все
            </button>


            <button
              className={
                filter === "failed"
                  ? "filter active"
                  : "filter"
              }
              onClick={() => {
                void changeFilter(
                  "failed"
                );
              }}
            >
              Неисправные
            </button>


            <button
              className={
                filter === "warning"
                  ? "filter active"
                  : "filter"
              }
              onClick={() => {
                void changeFilter(
                  "warning"
                );
              }}
            >
              Риск
            </button>


            <button
              className={
                filter === "anomaly"
                  ? "filter active"
                  : "filter"
              }
              onClick={() => {
                void changeFilter(
                  "anomaly"
                );
              }}
            >
              Аномалии
            </button>

          </div>


          <div className="sensor-list">

            {initialLoading ? (

              <div className="empty">
                Загрузка...
              </div>

            ) : sensors.length === 0 ? (

              <div className="empty">
                Датчики не найдены
              </div>

            ) : (

              sensors.map(
                (sensor) => (

                  <button
                    key={sensor.id}
                    className={
                      activeSensorId === sensor.id
                        ? "sensor-card selected"
                        : "sensor-card"
                    }
                    onClick={() => {
                      void selectSensor(
                        sensor.id
                      );
                    }}
                  >

                    <div className="sensor-card-top">

                      <div>

                        <strong>
                          {sensor.name}
                        </strong>

                        <span className="sensor-id">
                          ID {sensor.id}
                        </span>

                      </div>


                      <span
                        className={
                          sensor.currentState === "Неисправен"
                            ? "state state-fault"
                            : "state"
                        }
                      >
                        {sensor.currentState}
                      </span>

                    </div>


                    <div className="sensor-meta">

                      <span>
                        {sensor.type}
                      </span>

                      <span>
                        {sensor.objectName ??
                          "Объект не указан"}
                      </span>

                    </div>


                    <div className="sensor-signals">

                      {sensor.warning && (

                        <span className="signal warning">
                          Предупреждение
                        </span>
                      )}


                      {sensor.anomalyCandidate && (

                        <span className="signal anomaly">
                          Аномалия
                        </span>
                      )}


                      {!sensor.warning &&
                        !sensor.anomalyCandidate && (

                          <span className="signal neutral">
                            Без активных сигналов
                          </span>
                        )}

                    </div>

                  </button>
                )
              )
            )}

          </div>

        </section>


        {/* RIGHT */}

        <section className="details-panel">


          {/* Маленький индикатор загрузки.
              Контент не исчезает. */}

          {detailsLoading && (

            <div className="details-loading">

              <span className="loading-spinner" />

              Обновление

            </div>
          )}


          {!selectedSensor ? (

            <div className="details-placeholder">

              <div className="placeholder-icon">
                <SensorIcon />
              </div>

              <h2>
                Выберите датчик
              </h2>

              <p>
                Здесь появятся текущее состояние,
                история событий и результаты анализа.
              </p>

            </div>

          ) : (

            <>

              <div className="details-header">

                <div>

                  <span className="eyebrow">
                    Датчик {selectedSensor.id}
                  </span>

                  <h2>
                    {selectedSensor.name}
                  </h2>

                  <p>
                    {selectedSensor.objectName ??
                      "Объект не указан"}
                  </p>

                </div>


                <span
                  className={
                    currentState === "Неисправен"
                      ? "large-state fault"
                      : "large-state"
                  }
                >
                  {currentState}
                </span>

              </div>


              <div className="details-grid">

                <article className="info-card">

                  <span className="info-label">
                    Тип
                  </span>

                  <strong>
                    {selectedSensor.type}
                  </strong>

                </article>


                <article className="info-card">

                  <span className="info-label">
                    Последнее событие
                  </span>

                  <strong>
                    {formatDate(
                      selectedSensor.lastEventAt
                    )}
                  </strong>

                </article>


                <article className="info-card">

                  <span className="info-label">
                    Статус прогноза
                  </span>

                  <strong>
                    {predictionStatusText(
                      assessment?.predictionStatus ??
                      selectedSensor.predictionStatus
                    )}
                  </strong>

                </article>

              </div>


              <article className="risk-card">

                <div className="risk-header">

                  <div>

                    <span className="eyebrow">
                      Прогноз
                    </span>

                    <h3>
                      Индекс риска
                    </h3>

                  </div>


                  <strong className="risk-value">
                    {riskText(
                      currentRiskScore
                    )}
                  </strong>

                </div>


                {currentRiskScore !== null && (

                  <div className="risk-track">

                    <div
                      className="risk-fill"
                      style={{
                        width:
                          `${currentRiskScore * 100}%`,
                      }}
                    />

                  </div>
                )}


                <p className="muted">

                  Индекс отражает оценку модели,
                  а не подтверждённую вероятность
                  физической поломки.

                </p>

              </article>


              <article className="analysis-card">

                <div className="section-title">

                  <div>

                    <span className="eyebrow">
                      Анализ
                    </span>

                    <h3>
                      Факторы внимания
                    </h3>

                  </div>

                </div>


                {currentRiskFactors.length === 0 ? (

                  <p className="muted">
                    Выраженных факторов риска сейчас нет.
                  </p>

                ) : (

                  <div className="factor-list">

                    {currentRiskFactors.map(
                      (
                        factor,
                        index
                      ) => (

                        <div
                          className="factor"
                          key={
                            `${factor}-${index}`
                          }
                        >

                          <span className="factor-dot" />

                          {factor}

                        </div>
                      )
                    )}

                  </div>
                )}

              </article>


              <article className="history-card">

                <div className="section-title">

                  <div>

                    <span className="eyebrow">
                      Журнал
                    </span>

                    <h3>
                      Последние события
                    </h3>

                  </div>


                  <span className="history-count">
                    {history.length}
                  </span>

                </div>


                {history.length === 0 ? (

                  <p className="muted">
                    История отсутствует.
                  </p>

                ) : (

                  <div className="table-wrapper">

                    <table>

                      <thead>

                        <tr>
                          <th>Время</th>
                          <th>Состояние</th>
                          <th>Значение</th>
                          <th>Тревога</th>
                        </tr>

                      </thead>


                      <tbody>

                        {history.map(
                          (
                            event,
                            index
                          ) => (

                            <tr
                              key={
                                `${event.timestamp}-${index}`
                              }
                            >

                              <td>
                                {formatDate(
                                  event.timestamp
                                )}
                              </td>


                              <td>
                                {event.state}
                              </td>


                              <td>

                                {event.value === null
                                  ? "—"
                                  : `${event.value}${
                                      event.unit
                                        ? ` ${event.unit}`
                                        : ""
                                    }`}

                              </td>


                              <td>
                                {event.alarm
                                  ? "Да"
                                  : "Нет"}
                              </td>

                            </tr>
                          )
                        )}

                      </tbody>

                    </table>

                  </div>
                )}

              </article>

            </>
          )}

        </section>

      </main>

    </div>
  );
}

function SensorIcon({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 64 64"
      className={className}
      aria-hidden="true"
      fill="none"
    >
      <rect
        x="18"
        y="22"
        width="28"
        height="20"
        rx="8"
        stroke="currentColor"
        strokeWidth="2.8"
      />
      <circle
        cx="32"
        cy="32"
        r="4.5"
        stroke="currentColor"
        strokeWidth="2.8"
      />
      <path
        d="M32 18V13"
        stroke="currentColor"
        strokeWidth="2.8"
        strokeLinecap="round"
      />
      <path
        d="M24 46V51"
        stroke="currentColor"
        strokeWidth="2.8"
        strokeLinecap="round"
      />
      <path
        d="M32 46V53"
        stroke="currentColor"
        strokeWidth="2.8"
        strokeLinecap="round"
      />
      <path
        d="M40 46V51"
        stroke="currentColor"
        strokeWidth="2.8"
        strokeLinecap="round"
      />
      <path
        d="M24 20C25.8 16.9 28.6 15 32 15C35.4 15 38.2 16.9 40 20"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      <path
        d="M21 16C23.7 11.9 27.5 10 32 10C36.5 10 40.3 11.9 43 16"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        opacity="0.75"
      />
    </svg>
  );
}

export default App;