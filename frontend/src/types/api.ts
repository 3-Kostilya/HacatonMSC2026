export type PredictionStatus =
  | "scored"
  | "already_faulty"
  | "unknown_state"
  | "stale_observation"
  | "insufficient_history"
  | "not_available";

export type SensorGroup =
  | "healthy"
  | "warning"
  | "unavailable";

export interface SensorListItem {
  id: string;
  name: string;
  type: string;
  objectName: string | null;
  currentState: string;

  // Индекс относительно порога. Это не вероятность физической поломки.
  riskScore: number | null;

  warning: boolean | null;
  predictionStatus: PredictionStatus;
  anomalyCandidate: boolean;

  // Операционное правило R6.
  ruleScore: number | null;
  threshold: number | null;
  thresholdCrossed: boolean | null;
  mlPredictionStatus: string | null;

  // Исследовательская CatBoost-модель.
  researchScore: number | null;
  researchPredictionStatus: string | null;
}

export interface SensorDetails extends SensorListItem {
  lastEventAt: string | null;
  riskFactors: string[];
  objectId: string | null;
  sensorType: string | null;
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
  scoreContributions: Record<string, number> | null;

  researchScore: number | null;
  researchPredictionStatus: string | null;
  researchModelVersion: string | null;
  researchScoreKind: string | null;
}

export interface DashboardSummary {
  totalSensors: number;
  withoutWarnings: number;
  warnings: number;
  predictionUnavailable: number;

  // Оставлены в API для совместимости, но на дашборде не используются.
  registeredFaults?: number;
  anomalyCandidates?: number;
}

export interface HealthResponse {
  status: string;
}

export interface RawCapabilities {
  modelReady: boolean;
  referencesReady: boolean;
  acceptedJournalFormats: string[];
  acceptedReferenceFormats: string[];
}

export interface RawImportStart {
  batchId: string;
  status: string;
}

export interface RawImportStatus {
  batchId: string;
  filename: string;
  status: "queued" | "processing" | "processed" | "failed";
  stage: string;
  rowsCount: number;
  forecastsCount: number;
  sensorsCount?: number;
  errorMessage: string | null;
  updatedAt: string;
}
