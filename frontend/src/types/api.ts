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

  /*
   * Поле оставляем для совместимости backend,
   * но больше не выводим его пользователю
   * как вероятность.
   */
  riskScore: number | null;

  warning: boolean | null;

  predictionStatus: PredictionStatus;

  anomalyCandidate: boolean;

  /*
   * Актуальные показатели текущей ML-части,
   * которые backend отдаёт frontend.
   */
  ruleScore: number | null;

  threshold: number | null;

  thresholdCrossed: boolean | null;

  mlPredictionStatus: string | null;
}

export interface SensorDetails
  extends SensorListItem {

  lastEventAt: string | null;

  riskFactors: string[];

  objectId: string | null;

  sensorType: string | null;
}

export interface SensorEvent {
  timestamp: string;

  state: string;

  alarm: boolean;

  value:
    | string
    | number
    | null;

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

  ruleScore: number | null;

  threshold: number | null;

  thresholdCrossed: boolean | null;

  mlPredictionStatus: string | null;

  predictionTime: string | null;

  admissionStatus: string | null;

  admissionReason: string | null;

  unavailableReason: string | null;

  warningReason: string | null;

  policyVersion: string | null;

  scoreContributions:
    | Record<string, number>
    | null;
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