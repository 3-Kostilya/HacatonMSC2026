import type {
  DashboardSummary,
  SensorDetails,
  SensorListItem,
} from '../types/sensor'

const API_URL =
  import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

async function request<T>(url: string): Promise<T> {
  const response = await fetch(`${API_URL}${url}`)

  if (!response.ok) {
    throw new Error(`Ошибка API: ${response.status}`)
  }

  return response.json()
}

export function getSummary() {
  return request<DashboardSummary>('/api/dashboard/summary')
}

export function getFailedSensors() {
  return request<SensorListItem[]>(
    '/api/sensors?group=failed&limit=5',
  )
}

export function getCriticalSensors() {
  return request<SensorListItem[]>(
    '/api/sensors?group=critical&limit=5',
  )
}

export function getRiskSensors() {
  return request<SensorListItem[]>(
    '/api/sensors?group=risk&limit=5',
  )
}

export function searchSensors(query: string) {
  return request<SensorListItem[]>(
    `/api/sensors/search?q=${encodeURIComponent(query)}`,
  )
}

export function getSensor(id: string) {
  return request<SensorDetails>(
    `/api/sensors/${encodeURIComponent(id)}`,
  )
}