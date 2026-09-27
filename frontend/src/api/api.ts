import type {
  DashboardSummary,
  HealthResponse,
  SensorAssessment,
  SensorDetails,
  SensorEvent,
  SensorGroup,
  SensorListItem,
} from "../types/api";


const API_URL =
  (
    import.meta.env.VITE_API_URL ??
    "http://127.0.0.1:8000"
  ).replace(/\/+$/, "");


async function request<T>(
  path: string
): Promise<T> {

  let response: Response;


  try {

    response =
      await fetch(
        `${API_URL}${path}`
      );

  } catch {

    throw new Error(
      "Не удалось подключиться к backend"
    );
  }


  if (!response.ok) {

    let message =
      `Ошибка API: ${response.status}`;


    try {

      const data =
        await response.json();


      if (
        typeof data.detail === "string"
      ) {
        message =
          data.detail;
      }

    } catch {
      // Ответ не является JSON.
    }


    throw new Error(
      message
    );
  }


  return response.json();
}


export function getHealth() {

  return request<HealthResponse>(
    "/api/health"
  );
}


export function getDashboardSummary() {

  return request<DashboardSummary>(
    "/api/dashboard/summary"
  );
}


export function getSensors(
  group?: SensorGroup,
  limit = 100
) {

  const params =
    new URLSearchParams();

  params.set(
    "limit",
    String(limit)
  );


  if (group) {

    params.set(
      "group",
      group
    );
  }


  return request<SensorListItem[]>(
    `/api/sensors?${params.toString()}`
  );
}


export function searchSensors(
  query: string,
  limit = 50
) {

  const params =
    new URLSearchParams();

  params.set(
    "q",
    query
  );

  params.set(
    "limit",
    String(limit)
  );


  return request<SensorListItem[]>(
    `/api/sensors/search?${params.toString()}`
  );
}


export function getSensorDetails(
  sensorId: string
) {

  const id =
    encodeURIComponent(
      sensorId
    );


  return request<SensorDetails>(
    `/api/sensors/${id}`
  );
}


export function getSensorHistory(
  sensorId: string
) {

  const id =
    encodeURIComponent(
      sensorId
    );


  return request<SensorEvent[]>(
    `/api/sensors/${id}/history`
  );
}


export function getSensorAssessment(
  sensorId: string
) {

  const id =
    encodeURIComponent(
      sensorId
    );


  return request<SensorAssessment>(
    `/api/sensors/${id}/assessment`
  );
}