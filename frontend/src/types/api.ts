export type PredictionStatus =
  | "scored"
  | "already_faulty"
  | "unknown_state"
  | "stale_observation"
  | "insufficient_history"
  | "not_available";

export type SensorGroup =
  | "failed"
  | "warning"
  | "anomaly";


export interface SensorListItem {
  id: string;
  name: string;
  type: string;
  objectName: string | null;

  currentState: string;

  riskScore: number | null;
  warning: boolean | null;

  predictionStatus: PredictionStatus;

  anomalyCandidate: boolean;
}


export interface SensorDetails extends SensorListItem {
  lastEventAt: string | null;

  riskFactors: string[];
}


export interface SensorEvent {
  timestamp: string;

  state: string;

  alarm: boolean;

  value: string | number | null;

  unit: string | null;
}


export interface SensorAssessment {
  id: string;

  currentState: string;

  riskScore: number | null;

  warning: boolean | null;

  predictionStatus: PredictionStatus;

  anomalyCandidate: boolean;

  riskFactors: string[];
}


export interface DashboardSummary {
  totalSensors: number;

  registeredFaults: number;

  warnings: number;

  anomalyCandidates: number;

  predictionUnavailable: number;
}


export interface HealthResponse {
  status: string;
}