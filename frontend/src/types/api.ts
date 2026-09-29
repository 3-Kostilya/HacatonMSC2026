export type PredictionStatus =
  | "scored"
  | "already_faulty"
  | "unknown_state"
  | "stale_observation"
  | "insufficient_history"
  | "not_available";

export type SensorGroup =
  | "failed"
  | "warning";

export interface SensorListItem {
  id: string;
  name: string;
  type: string;
  objectName: string | null;
  currentState: string;

  // Совместимость с backend. Это индекс относительно порога, а не вероятность.
  riskScore: number | null;

  warning: boolean | null;
  predictionStatus: PredictionStatus;

  // Балл Round 7 и результат статических ворот.
  ruleScore: number | null;
  threshold: number | null;
  thresholdCrossed: boolean | null;
  mlPredictionStatus: string | null;

  // Итоговая исследовательская оценка Round 7.
  // Score не трактуется как вероятность физической поломки.
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
  registeredFaults: number;
  warnings: number;
  predictionUnavailable: number;
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
