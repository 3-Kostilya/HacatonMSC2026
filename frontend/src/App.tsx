import { useEffect, useState } from 'react'
import './App.css'

import {
  getCriticalSensors,
  getFailedSensors,
  getRiskSensors,
  getSensor,
  getSummary,
  searchSensors,
} from './api/sensors'

import type {
  DashboardSummary,
  SensorDetails,
  SensorListItem,
} from './types/sensor'

function formatProbability(value: number | null) {
  if (value === null) return '—'

  return `${(value * 100).toFixed(1)}%`
}

function SensorList({
  title,
  sensors,
  onSelect,
}: {
  title: string
  sensors: SensorListItem[]
  onSelect: (sensor: SensorListItem) => void
}) {
  return (
    <section className="sensor-panel">
      <div className="panel-title">
        <h2>{title}</h2>
        <span>{sensors.length}</span>
      </div>

      {sensors.length === 0 ? (
        <div className="empty">Нет данных</div>
      ) : (
        sensors.map((sensor) => (
          <button
            key={sensor.id}
            className="sensor-row"
            onClick={() => onSelect(sensor)}
          >
            <div>
              <strong>{sensor.name}</strong>

              <span>
                {sensor.type}
                {sensor.objectName
                  ? ` · ${sensor.objectName}`
                  : ''}
              </span>
            </div>

            <div className="sensor-risk">
              {formatProbability(
                sensor.failureProbability,
              )}

              <small>{sensor.status}</small>
            </div>
          </button>
        ))
      )}
    </section>
  )
}

function App() {
  const [summary, setSummary] =
    useState<DashboardSummary | null>(null)

  const [failed, setFailed] =
    useState<SensorListItem[]>([])

  const [critical, setCritical] =
    useState<SensorListItem[]>([])

  const [risk, setRisk] =
    useState<SensorListItem[]>([])

  const [selectedSensor, setSelectedSensor] =
    useState<SensorDetails | null>(null)

  const [search, setSearch] = useState('')

  const [searchResults, setSearchResults] =
    useState<SensorListItem[]>([])

  const [loading, setLoading] = useState(true)

  const [error, setError] =
    useState<string | null>(null)

  useEffect(() => {
    async function loadDashboard() {
      try {
        setLoading(true)

        const [
          summaryData,
          failedData,
          criticalData,
          riskData,
        ] = await Promise.all([
          getSummary(),
          getFailedSensors(),
          getCriticalSensors(),
          getRiskSensors(),
        ])

        setSummary(summaryData)
        setFailed(failedData)
        setCritical(criticalData)
        setRisk(riskData)
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : 'Не удалось загрузить данные',
        )
      } finally {
        setLoading(false)
      }
    }

    loadDashboard()
  }, [])

  useEffect(() => {
    if (!search.trim()) {
      setSearchResults([])
      return
    }

    const timeout = setTimeout(async () => {
      try {
        const result = await searchSensors(search)
        setSearchResults(result)
      } catch {
        setSearchResults([])
      }
    }, 300)

    return () => clearTimeout(timeout)
  }, [search])

  async function selectSensor(
    sensor: SensorListItem,
  ) {
    try {
      const details = await getSensor(sensor.id)
      setSelectedSensor(details)
      setSearch('')
      setSearchResults([])
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Не удалось загрузить датчик',
      )
    }
  }

  if (loading) {
    return (
      <div className="center-message">
        Загрузка данных...
      </div>
    )
  }

  if (error && !summary) {
    return (
      <div className="center-message error">
        {error}
      </div>
    )
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <span className="caption">
            СМВУ / PREDICTIVE MONITORING
          </span>

          <h1>
            Система прогнозирования отказов датчиков
          </h1>

          <p>
            Мониторинг состояния и выявление
            признаков деградации оборудования
          </p>
        </div>

        <div className="online">
          <span />
          Система подключена
        </div>
      </header>

      {summary && (
        <section className="stats">
          <div className="stat-card">
            <span>Всего датчиков</span>
            <strong>{summary.total}</strong>
          </div>

          <div className="stat-card">
            <span>Под риском</span>
            <strong>{summary.atRisk}</strong>
          </div>

          <div className="stat-card">
            <span>Критичных</span>
            <strong>{summary.critical}</strong>
          </div>

          <div className="stat-card">
            <span>Неисправно</span>
            <strong>{summary.failed}</strong>
          </div>
        </section>
      )}

      <section className="lists">
        <SensorList
          title="Неисправные"
          sensors={failed}
          onSelect={selectSensor}
        />

        <SensorList
          title="Критичные"
          sensors={critical}
          onSelect={selectSensor}
        />

        <SensorList
          title="Под риском"
          sensors={risk}
          onSelect={selectSensor}
        />
      </section>

      <section className="search-section">
        <span className="caption">
          ПОИСК ДАТЧИКА
        </span>

        <h2>
          Найти конкретное устройство
        </h2>

        <input
          value={search}
          onChange={(event) =>
            setSearch(event.target.value)
          }
          placeholder="ID, название, тип или объект"
        />

        {searchResults.length > 0 && (
          <div className="search-results">
            {searchResults.map((sensor) => (
              <button
                key={sensor.id}
                onClick={() =>
                  selectSensor(sensor)
                }
              >
                <div>
                  <strong>{sensor.name}</strong>

                  <span>
                    {sensor.type}
                    {sensor.objectName
                      ? ` · ${sensor.objectName}`
                      : ''}
                  </span>
                </div>

                <span>
                  {formatProbability(
                    sensor.failureProbability,
                  )}
                </span>
              </button>
            ))}
          </div>
        )}
      </section>

      {selectedSensor && (
        <section className="details">
          <div className="details-header">
            <div>
              <span className="caption">
                ВЫБРАННЫЙ ДАТЧИК
              </span>

              <h2>{selectedSensor.name}</h2>

              <p>
                {selectedSensor.type}
                {selectedSensor.objectName
                  ? ` · ${selectedSensor.objectName}`
                  : ''}
              </p>
            </div>

            <div className="prediction">
              <span>Риск отказа</span>

              <strong>
                {formatProbability(
                  selectedSensor.failureProbability,
                )}
              </strong>
            </div>
          </div>

          <div className="metrics">
            {selectedSensor.metrics.map(
              (metric) => (
                <div
                  className="metric"
                  key={metric.key}
                >
                  <span>{metric.label}</span>

                  <strong>
                    {metric.value ?? '—'}
                    {metric.unit
                      ? ` ${metric.unit}`
                      : ''}
                  </strong>
                </div>
              ),
            )}
          </div>

          {selectedSensor.riskFactors.length >
            0 && (
            <div className="risk-factors">
              <span className="caption">
                ФАКТОРЫ РИСКА
              </span>

              <ul>
                {selectedSensor.riskFactors.map(
                  (factor) => (
                    <li key={factor}>
                      {factor}
                    </li>
                  ),
                )}
              </ul>
            </div>
          )}
        </section>
      )}
    </div>
  )
}

export default App