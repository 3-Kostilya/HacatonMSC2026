import type {
  DashboardSummary,
  HealthResponse,
  RawCapabilities,
  RawImportStart,
  RawImportStatus,
  SensorAssessment,
  SensorDetails,
  SensorEvent,
  SensorGroup,
  SensorListItem,
} from "../types/api";

const API_URL = (import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000").replace(
  /\/+$/,
  "",
);

async function readError(response: Response) {
  let message = `Ошибка API: ${response.status}`;

  try {
    const text = await response.text();
    if (!text) return message;

    try {
      const data = JSON.parse(text) as { detail?: unknown };
      if (typeof data.detail === "string") return data.detail;
    } catch {
      return text.slice(0, 500);
    }
  } catch {
    // Оставляем стандартное сообщение.
  }

  return message;
}

async function request<T>(path: string): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_URL}${path}`);
  } catch {
    throw new Error("Не удалось подключиться к backend");
  }

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  return response.json() as Promise<T>;
}

export function getHealth() {
  return request<HealthResponse>("/api/health");
}

export function getDashboardSummary() {
  return request<DashboardSummary>("/api/dashboard/summary");
}

export function getSensors(group?: SensorGroup) {
  const params = new URLSearchParams();
  if (group) params.set("group", group);

  const query = params.toString();
  return request<SensorListItem[]>(`/api/sensors${query ? `?${query}` : ""}`);
}

export function searchSensors(query: string) {
  const params = new URLSearchParams();
  params.set("q", query);
  return request<SensorListItem[]>(`/api/sensors/search?${params.toString()}`);
}

export function getSensorDetails(sensorId: string) {
  return request<SensorDetails>(`/api/sensors/${encodeURIComponent(sensorId)}`);
}

export function getSensorHistory(sensorId: string) {
  return request<SensorEvent[]>(
    `/api/sensors/${encodeURIComponent(sensorId)}/history`,
  );
}

export function getSensorAssessment(sensorId: string) {
  return request<SensorAssessment>(
    `/api/sensors/${encodeURIComponent(sensorId)}/assessment`,
  );
}

export function getRawCapabilities() {
  return request<RawCapabilities>("/api/raw/capabilities");
}

export async function uploadRawData(
  journal: File,
  channels?: File | null,
  objects?: File | null,
) {
  const form = new FormData();
  form.append("journal", journal);
  if (channels) form.append("channels", channels);
  if (objects) form.append("objects", objects);

  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/raw/import`, {
      method: "POST",
      body: form,
    });
  } catch {
    throw new Error("Не удалось подключиться к backend");
  }

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  return response.json() as Promise<RawImportStart>;
}

export function getRawImportStatus(batchId: string) {
  return request<RawImportStatus>(
    `/api/raw/import/${encodeURIComponent(batchId)}`,
  );
}
