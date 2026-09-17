export interface DashboardSummary {
  total: number
  atRisk: number
  critical: number
  failed: number
}

export interface SensorListItem {
  id: string
  name: string
  type: string
  objectName: string | null
  status: string
  failureProbability: number | null
}

export interface SensorMetric {
  key: string
  label: string
  value: string | number | null
  unit?: string | null
}

export interface SensorDetails extends SensorListItem {
  metrics: SensorMetric[]
  riskFactors: string[]
}