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
  getRawCapabilities,
  getRawImportStatus,
  uploadRawData,
} from "./api/api";

import type {
  DashboardSummary,
  SensorAssessment,
  SensorDetails,
  SensorEvent,
  SensorGroup,
  SensorListItem,
  RawCapabilities,
  RawImportStatus,
} from "./types/api";

import "./App.css";

type Filter = "all" | SensorGroup;
type Theme = "dark" | "light";

const CONTRIBUTION_LABELS: Record<string, string> = {
  registered_fault_text_count_24h: "Записи «Неисправен» за 24 часа",
  registered_fault_text_count_168h: "Записи «Неисправен» за 7 дней",
  completed_episode_count_168h: "Завершённые проблемные эпизоды за 7 дней",
  technical_message_count_24h: "Технические сообщения за 24 часа",
};

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor">
      <circle cx="12" cy="12" r="3.5" />
      <path d="M12 2.5V5 M12 19V21.5 M4.5 4.5L6.3 6.3 M17.7 17.7L19.5 19.5 M2.5 12H5 M19 12H21.5 M4.5 19.5L6.3 17.7 M17.7 6.3L19.5 4.5" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor">
      <path d="M20.2 15.2 A8.6 8.6 0 0 1 8.8 3.8 A8.6 8.6 0 1 0 20.2 15.2 Z" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor">
      <path d="M7 7L17 17" />
      <path d="M17 7L7 17" />
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor">
      <path d="M20 6V11H15" />
      <path d="M18.2 8 A7.5 7.5 0 1 0 19 15" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor">
      <path d="M12 16V4" />
      <path d="M7.5 8.5L12 4l4.5 4.5" />
      <path d="M5 14v4.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V14" />
    </svg>
  );
}

function SensorIcon({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" className={className} aria-hidden="true" fill="none">
      <rect x="18" y="22" width="28" height="20" rx="8" stroke="currentColor" strokeWidth="2.8" />
      <circle cx="32" cy="32" r="4.5" stroke="currentColor" strokeWidth="2.8" />
      <path d="M32 18V13" stroke="currentColor" strokeWidth="2.8" strokeLinecap="round" />
      <path d="M24 46V51 M32 46V53 M40 46V51" stroke="currentColor" strokeWidth="2.8" strokeLinecap="round" />
      <path d="M24 20C25.8 16.9 28.6 15 32 15C35.4 15 38.2 16.9 40 20" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
      <path d="M21 16C23.7 11.9 27.5 10 32 10C36.5 10 40.3 11.9 43 16" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" opacity="0.75" />
    </svg>
  );
}

function formatDate(value: string | null) {
  if (!value) return "Нет данных";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function predictionStatusText(status: string | null) {
  const names: Record<string, string> = {
    scored: "Прогноз рассчитан",
    scored_research: "ML-оценка рассчитана",
    already_faulty: "Уже неисправен",
    unknown_state: "Неизвестное состояние",
    stale_observation: "Данные устарели",
    insufficient_history: "Недостаточно истории",
    not_available: "Прогноз недоступен",
  };
  if (!status) return "Нет данных";
  return names[status] ?? status;
}

function scoreText(score: number | null, digits = 2) {
  if (score === null || Number.isNaN(score)) return "—";
  return score.toLocaleString("ru-RU", {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

function isFaultState(state: string) {
  return state.toLocaleLowerCase("ru-RU").includes("неисправ");
}

function displayOperationalState(state: string) {
  const technicalFallbacks = new Set([
    "Под риском",
    "Без прогноза",
    "Прогноз рассчитан",
    "Аномалия",
  ]);

  if (!state || technicalFallbacks.has(state)) {
    return "История не загружена";
  }

  return state;
}

function getInitialTheme(): Theme {
  const saved = localStorage.getItem("theme");
  return saved === "light" || saved === "dark" ? saved : "dark";
}

function App() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [backendOnline, setBackendOnline] = useState(false);
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [sensors, setSensors] = useState<SensorListItem[]>([]);
  const [selectedSensor, setSelectedSensor] = useState<SensorDetails | null>(null);
  const [activeSensorId, setActiveSensorId] = useState<string | null>(null);
  const [assessment, setAssessment] = useState<SensorAssessment | null>(null);
  const [history, setHistory] = useState<SensorEvent[]>([]);
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const [initialLoading, setInitialLoading] = useState(true);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [journalFile, setJournalFile] = useState<File | null>(null);
  const [channelsFile, setChannelsFile] = useState<File | null>(null);
  const [objectsFile, setObjectsFile] = useState<File | null>(null);
  const [rawCapabilities, setRawCapabilities] = useState<RawCapabilities | null>(null);
  const [rawImport, setRawImport] = useState<RawImportStatus | null>(null);

  const sensorRequestId = useRef(0);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", theme === "dark" ? "#15191e" : "#dde2e6");
  }, [theme]);

  async function loadOverview(selectedFilter: Filter) {
    const group = selectedFilter === "all" ? undefined : selectedFilter;
    const [dashboardData, sensorData] = await Promise.all([
      getDashboardSummary(),
      getSensors(group),
    ]);
    setSummary(dashboardData);
    setSensors(sensorData);
  }

  useEffect(() => {
    let cancelled = false;

    async function initialize() {
      try {
        setInitialLoading(true);
        setError(null);
        await getHealth();
        if (cancelled) return;
        setBackendOnline(true);

        const [dashboardData, sensorData] = await Promise.all([
          getDashboardSummary(),
          getSensors(),
        ]);

        if (cancelled) return;
        setSummary(dashboardData);
        setSensors(sensorData);
      } catch (err) {
        if (cancelled) return;
        setBackendOnline(false);
        setError(err instanceof Error ? err.message : "Не удалось подключиться к backend");
      } finally {
        if (!cancelled) setInitialLoading(false);
      }
    }

    void initialize();
    return () => {
      cancelled = true;
    };
  }, []);

  async function selectSensor(sensorId: string) {
    const requestId = ++sensorRequestId.current;
    setActiveSensorId(sensorId);
    setDetailsLoading(true);
    setError(null);

    try {
      const [sensorData, historyData, assessmentData] = await Promise.all([
        getSensorDetails(sensorId),
        getSensorHistory(sensorId),
        getSensorAssessment(sensorId),
      ]);

      if (requestId !== sensorRequestId.current) return;
      setSelectedSensor(sensorData);
      setHistory(historyData);
      setAssessment(assessmentData);
    } catch (err) {
      if (requestId !== sensorRequestId.current) return;
      setError(err instanceof Error ? err.message : "Не удалось загрузить данные датчика");
    } finally {
      if (requestId === sensorRequestId.current) setDetailsLoading(false);
    }
  }

  async function changeFilter(newFilter: Filter) {
    try {
      setError(null);
      setFilter(newFilter);
      setSearch("");
      const group = newFilter === "all" ? undefined : newFilter;
      setSensors(await getSensors(group));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось загрузить датчики");
    }
  }

  async function handleSearch(event: FormEvent) {
    event.preventDefault();
    const query = search.trim();

    try {
      setError(null);
      if (!query) {
        await loadOverview(filter);
        return;
      }
      setSensors(await searchSensors(query));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка поиска");
    }
  }

  async function refresh() {
    if (refreshing) return;

    try {
      setRefreshing(true);
      setError(null);
      await getHealth();
      setBackendOnline(true);

      if (search.trim()) {
        const [sensorData, dashboardData] = await Promise.all([
          searchSensors(search.trim()),
          getDashboardSummary(),
        ]);
        setSensors(sensorData);
        setSummary(dashboardData);
      } else {
        await loadOverview(filter);
      }

      if (selectedSensor) await selectSensor(selectedSensor.id);
    } catch (err) {
      setBackendOnline(false);
      setError(err instanceof Error ? err.message : "Не удалось обновить данные");
    } finally {
      setRefreshing(false);
    }
  }

  async function openUpload() {
    setUploadOpen(true);
    setRawImport(null);
    try {
      setRawCapabilities(await getRawCapabilities());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось проверить готовность загрузки");
    }
  }

  async function startRawUpload(event: FormEvent) {
    event.preventDefault();
    if (!journalFile || uploading) return;
    if ((channelsFile && !objectsFile) || (!channelsFile && objectsFile)) {
      setError("Справочник каналов и справочник объектов нужно загружать вместе");
      return;
    }

    try {
      setUploading(true);
      setError(null);
      const started = await uploadRawData(journalFile, channelsFile, objectsFile);
      let status = await getRawImportStatus(started.batchId);
      setRawImport(status);

      while (status.status === "queued" || status.status === "processing") {
        await new Promise((resolve) => window.setTimeout(resolve, 1200));
        status = await getRawImportStatus(started.batchId);
        setRawImport(status);
      }

      if (status.status === "failed") {
        throw new Error(status.errorMessage || "Обработка данных завершилась с ошибкой");
      }

      setRawCapabilities(await getRawCapabilities());
      await loadOverview("all");
      setFilter("all");
      setSearch("");
      setJournalFile(null);
      setChannelsFile(null);
      setObjectsFile(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось обработать данные");
    } finally {
      setUploading(false);
    }
  }

  const currentRuleScore = assessment?.ruleScore ?? selectedSensor?.ruleScore ?? null;
  const currentThreshold = assessment?.threshold ?? selectedSensor?.threshold ?? null;
  const currentThresholdCrossed = assessment?.thresholdCrossed ?? selectedSensor?.thresholdCrossed ?? null;
  const currentWarning = assessment?.warning ?? selectedSensor?.warning ?? false;
  const currentPredictionStatus = assessment?.predictionStatus ?? selectedSensor?.predictionStatus ?? "not_available";
  const currentPredictionTime = assessment?.predictionTime ?? null;
  const currentUnavailableReason = assessment?.unavailableReason ?? null;
  const currentAdmissionReason = assessment?.admissionReason ?? null;
  const currentState = displayOperationalState(assessment?.currentState ?? selectedSensor?.currentState ?? "");
  const currentResearchScore = assessment?.researchScore ?? selectedSensor?.researchScore ?? null;
  const currentResearchStatus = assessment?.researchPredictionStatus ?? selectedSensor?.researchPredictionStatus ?? null;

  const contributionEntries = Object.entries(assessment?.scoreContributions ?? {})
    .filter(([, value]) => Number.isFinite(value) && value !== 0)
    .map(([key, value]) => ({
      key,
      label: CONTRIBUTION_LABELS[key] ?? key,
      value,
    }));

  const r6Available = currentPredictionStatus === "scored" && currentRuleScore !== null;
  const r6Result = !r6Available
    ? "Недоступен"
    : currentThresholdCrossed === true
      ? "Порог превышен"
      : currentThresholdCrossed === false
        ? "Порог не превышен"
        : "Нет решения";

  const r6ResultClass = !r6Available
    ? "forecast-result unknown"
    : currentThresholdCrossed === true
      ? "forecast-result warning"
      : currentThresholdCrossed === false
        ? "forecast-result ok"
        : "forecast-result unknown";

  const systemDecision = currentWarning
    ? "Теневой сигнал"
    : currentPredictionStatus !== "scored"
      ? "Без прогноза"
      : "Порог не достигнут";

  const systemDecisionClass = currentWarning
    ? "decision-badge warning"
    : currentPredictionStatus !== "scored"
      ? "decision-badge unavailable"
      : "decision-badge ok";

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-mark">
            <img src="/logo.png" alt="Логотип системы" className="brand-logo" />
          </div>
          <div className="brand-text">
            <h1>Мониторинг инженерной инфраструктуры</h1>
            <p>Состояние каналов и исследовательский прогноз записи «Неисправен» на 24 часа</p>
          </div>
        </div>

        <div className="header-actions">
          <div className={backendOnline ? "connection online" : "connection offline"}>
            <span className="connection-dot" />
            <span>{backendOnline ? "Backend подключён" : "Нет соединения"}</span>
          </div>

          <button className="secondary-button upload-button" onClick={() => void openUpload()}>
            <span className="upload-icon"><UploadIcon /></span>
            <span>Загрузить данные</span>
          </button>

          <button
            className="icon-button theme-button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label={theme === "dark" ? "Включить светлую тему" : "Включить тёмную тему"}
            title={theme === "dark" ? "Светлая тема" : "Тёмная тема"}
          >
            {theme === "dark" ? <SunIcon /> : <MoonIcon />}
          </button>

          <button
            className={refreshing ? "secondary-button refreshing" : "secondary-button"}
            onClick={() => void refresh()}
            disabled={refreshing}
          >
            <span className="refresh-icon"><RefreshIcon /></span>
            <span>Обновить</span>
          </button>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button className="error-close" onClick={() => setError(null)} aria-label="Закрыть сообщение">
            <CloseIcon />
          </button>
        </div>
      )}

      <section className="summary-grid">
        <article className="summary-card">
          <span>Всего датчиков</span>
          <strong>{summary?.totalSensors ?? "—"}</strong>
        </article>
        <article className="summary-card">
          <span>Записи «Неисправен»</span>
          <strong>{summary?.registeredFaults ?? "—"}</strong>
        </article>
        <article className="summary-card">
          <span>Теневые сигналы</span>
          <strong>{summary?.warnings ?? "—"}</strong>
        </article>
        <article className="summary-card">
          <span>Без прогноза</span>
          <strong>{summary?.predictionUnavailable ?? "—"}</strong>
        </article>
      </section>

      <main className="workspace">
        <section className="sensor-panel">
          <div className="panel-header">
            <div>
              <h2>Датчики</h2>
              <p>Показано: {sensors.length}</p>
            </div>
          </div>

          <form className="search" onSubmit={(event) => void handleSearch(event)}>
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="ID, название, тип или объект"
            />
            <button type="submit">Найти</button>
          </form>

          <div className="filters">
            <button className={filter === "all" ? "filter active" : "filter"} onClick={() => void changeFilter("all")}>Все</button>
            <button className={filter === "failed" ? "filter active" : "filter"} onClick={() => void changeFilter("failed")}>«Неисправен»</button>
            <button className={filter === "warning" ? "filter active" : "filter"} onClick={() => void changeFilter("warning")}>Теневые сигналы</button>
          </div>

          <div className="sensor-list">
            {initialLoading ? (
              <div className="empty">Загрузка...</div>
            ) : sensors.length === 0 ? (
              <div className="empty">Датчики не найдены</div>
            ) : (
              sensors.map((sensor) => {
                const state = displayOperationalState(sensor.currentState);
                const noForecast = sensor.predictionStatus !== "scored";

                return (
                  <button
                    key={sensor.id}
                    className={activeSensorId === sensor.id ? "sensor-card selected" : "sensor-card"}
                    onClick={() => void selectSensor(sensor.id)}
                  >
                    <div className="sensor-card-top">
                      <div>
                        <strong>{sensor.name}</strong>
                        <span className="sensor-id">ID {sensor.id}</span>
                      </div>
                      <span className={isFaultState(state) ? "state state-fault" : "state"}>{state}</span>
                    </div>

                    <div className="sensor-meta">
                      <span>{sensor.type}</span>
                      <span>{sensor.objectName ?? "Объект не указан"}</span>
                    </div>

                    <div className="sensor-signals">
                      {sensor.warning ? (
                        <span className="signal warning">Теневой сигнал</span>
                      ) : noForecast ? (
                        <span className="signal unavailable">Без прогноза</span>
                      ) : (
                        <span className="signal neutral">Порог не достигнут</span>
                      )}
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </section>

        <section className="details-panel">
          {detailsLoading && (
            <div className="details-loading">
              <span className="loading-spinner" />
              Обновление
            </div>
          )}

          {!selectedSensor ? (
            <div className="details-placeholder">
              <div className="placeholder-icon"><SensorIcon /></div>
              <h2>Выберите датчик</h2>
              <p>Здесь появятся фактическое состояние, прогноз модели и последние события датчика.</p>
            </div>
          ) : (
            <>
              <div className="details-header">
                <div>
                  <span className="eyebrow">Датчик {selectedSensor.id}</span>
                  <h2>{selectedSensor.name}</h2>
                  <p>{selectedSensor.objectName ?? "Объект не указан"}</p>
                </div>
                <span className={isFaultState(currentState) ? "large-state fault" : "large-state"}>{currentState}</span>
              </div>

              <div className="details-grid">
                <article className="info-card">
                  <span className="info-label">Тип</span>
                  <strong>{selectedSensor.type}</strong>
                </article>
                <article className="info-card">
                  <span className="info-label">Последнее событие</span>
                  <strong>{formatDate(selectedSensor.lastEventAt)}</strong>
                </article>
                <article className="info-card">
                  <span className="info-label">Решение системы</span>
                  <strong className={systemDecisionClass}>{systemDecision}</strong>
                </article>
              </div>

              <article className="risk-card">
                <div className="risk-header">
                  <div>
                    <span className="eyebrow">Исследовательский ML-балл на 24 часа</span>
                    <h3>CatBoost</h3>
                  </div>
                  <strong className="risk-value">{scoreText(currentResearchScore, 3)}</strong>
                </div>

                <div className="forecast-grid">
                  <div className="forecast-item">
                    <span className="forecast-label">Статус ML</span>
                    <strong>{predictionStatusText(currentResearchStatus)}</strong>
                  </div>
                  <div className="forecast-item">
                    <span className="forecast-label">Решение системы</span>
                    <strong className={systemDecisionClass}>{systemDecision}</strong>
                  </div>
                  <div className="forecast-item">
                    <span className="forecast-label">Последний расчёт</span>
                    <strong>{formatDate(currentPredictionTime)}</strong>
                  </div>
                </div>

                <p className="forecast-note">
                  ML-score показывает оценку модели для появления нового зарегистрированного проблемного эпизода в ближайшие 24 часа. Это не калиброванная вероятность физической поломки.
                </p>

                {currentUnavailableReason && (
                  <p className="forecast-note"><strong>Почему прогноз недоступен:</strong> {currentUnavailableReason}</p>
                )}

                {!currentUnavailableReason && currentAdmissionReason && currentPredictionStatus !== "scored" && (
                  <p className="forecast-note"><strong>Причина статуса:</strong> {currentAdmissionReason}</p>
                )}
              </article>

              <article className="analysis-card">
                <div className="section-title">
                  <div>
                    <span className="eyebrow">Теневой контрольный сигнал</span>
                    <h3>Индекс R6 по истории событий</h3>
                  </div>
                  <span className={currentWarning ? "signal warning" : "signal neutral"}>
                    {currentWarning ? "Теневой сигнал" : "Порог не достигнут"}
                  </span>
                </div>

                <div className="forecast-grid compact-grid">
                  <div className="forecast-item">
                    <span className="forecast-label">Индекс</span>
                    <strong>{scoreText(currentRuleScore)}</strong>
                  </div>
                  <div className="forecast-item">
                    <span className="forecast-label">Порог</span>
                    <strong>{scoreText(currentThreshold)}</strong>
                  </div>
                  <div className="forecast-item">
                    <span className="forecast-label">Результат</span>
                    <strong className={r6ResultClass}>{r6Result}</strong>
                  </div>
                </div>

                {contributionEntries.length === 0 ? (
                  <p className="muted">Выраженные факторы индекса не выделены.</p>
                ) : (
                  <div className="factor-list">
                    {contributionEntries.map((factor) => (
                      <div className="factor" key={factor.key}>
                        <span className="factor-dot" />
                        <span>{factor.label}</span>
                        <strong className="factor-value">вклад {scoreText(factor.value)}</strong>
                      </div>
                    ))}
                  </div>
                )}
              </article>

              <details className="history-card history-details">
                <summary className="history-summary">
                  <div>
                    <span className="eyebrow">Журнал</span>
                    <h3>Последние события</h3>
                  </div>
                  <div className="history-summary-meta">
                    <span className="history-count">{history.length}</span>
                    <span className="history-toggle">Показать</span>
                  </div>
                </summary>

                {history.length === 0 ? (
                  <p className="muted">История событий пока не загружена.</p>
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
                        {history.map((event, index) => (
                          <tr key={`${event.timestamp}-${index}`}>
                            <td>{formatDate(event.timestamp)}</td>
                            <td>{event.state}</td>
                            <td>{event.value === null ? "—" : `${event.value}${event.unit ? ` ${event.unit}` : ""}`}</td>
                            <td>{event.alarm ? "Да" : "Нет"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </details>
            </>
          )}
        </section>
      </main>

      {uploadOpen && (
        <div className="modal-backdrop" onMouseDown={() => !uploading && setUploadOpen(false)}>
          <section className="upload-modal" onMouseDown={(event) => event.stopPropagation()}>
            <div className="upload-modal-header">
              <div>
                <span className="eyebrow">ДАННЫЕ</span>
                <h2>Загрузка и обработка данных</h2>
              </div>
              <button className="error-close" onClick={() => !uploading && setUploadOpen(false)} aria-label="Закрыть">
                <CloseIcon />
              </button>
            </div>

            <p className="upload-description">
              Загрузите журнал событий для анализа состояния датчиков и формирования
              прогноза. При необходимости обновите справочники каналов и объектов.
            </p>

            <div className="runtime-status">
              <span className={rawCapabilities?.referencesReady ? "runtime-dot ready" : "runtime-dot"} />
              <span>{rawCapabilities?.referencesReady ? "Справочники сохранены" : "Для первой загрузки добавьте два справочника"}</span>
              <span className={rawCapabilities?.modelReady ? "runtime-dot ready" : "runtime-dot"} />
              <span>{rawCapabilities?.modelReady ? "ML-модель готова" : "ML-модель не найдена"}</span>
            </div>

            <form className="upload-form" onSubmit={(event) => void startRawUpload(event)}>
              <label className="file-field primary-file">
                <span>Журнал событий *</span>
                <small>CSV-файл журнала событий или архив 7z</small>
                <input type="file" accept=".csv,.7z" onChange={(event) => setJournalFile(event.target.files?.[0] ?? null)} disabled={uploading} />
                <strong>{journalFile?.name ?? "Файл не выбран"}</strong>
              </label>

              <div className="reference-grid">
                <label className="file-field">
                  <span>Справочник каналов</span>
                  <small>Используется для определения типа датчика и его принадлежности к объекту</small>
                  <input type="file" accept=".csv" onChange={(event) => setChannelsFile(event.target.files?.[0] ?? null)} disabled={uploading} />
                  <strong>{channelsFile?.name ?? "Использовать сохранённый"}</strong>
                </label>

                <label className="file-field">
                  <span>Справочник объектов</span>
                  <small>Содержит сведения об объектах инженерной инфраструктуры</small>
                  <input type="file" accept=".csv" onChange={(event) => setObjectsFile(event.target.files?.[0] ?? null)} disabled={uploading} />
                  <strong>{objectsFile?.name ?? "Использовать сохранённый"}</strong>
                </label>
              </div>

              {rawImport && (
                <div className={rawImport.status === "failed" ? "upload-progress failed" : rawImport.status === "processed" ? "upload-progress done" : "upload-progress"}>
                  <div>
                    <strong>{rawImport.stage}</strong>
                    <span>{rawImport.filename}</span>
                  </div>
                  {rawImport.status === "processed" && (
                    <span>{rawImport.rowsCount.toLocaleString("ru-RU")} событий · {rawImport.forecastsCount.toLocaleString("ru-RU")} прогнозов</span>
                  )}
                  {rawImport.errorMessage && <span>{rawImport.errorMessage}</span>}
                </div>
              )}

              <div className="upload-actions">
                <button type="button" className="secondary-button" onClick={() => setUploadOpen(false)} disabled={uploading}>Отмена</button>
                <button type="submit" className="primary-button" disabled={!journalFile || uploading}>
                  {uploading ? "Обработка..." : "Загрузить и обработать"}
                </button>
              </div>
            </form>
          </section>
        </div>
      )}
    </div>
  );
}

export default App;
