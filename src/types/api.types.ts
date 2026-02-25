// ─── Device Snapshot ─────────────────────────────────────────────────────────
// Real field schema from the device monitoring agent (33 fields).

export interface DeviceSnapshot {
  agent_id: string;
  time: string;
  os: string;
  os_version: string;
  memory_size_gb: number;
  cpu_count: number;
  memory_count: number;
  uptime_days: number;
  p90_memory_utilization: number;
  avg_memory_utilization: number;
  max_memory_usage: number;
  p90_processor_time: number;
  avg_processor_time: number;
  max_cpu_usage: number;
  avg_cpu_temp: number;
  p90_cpu_temp: number;
  avg_boot_time: number;
  p90_boot_time: number;
  total_app_crash: number;
  p10_battery_health: number;
  avg_battery_health: number;
  avg_battery_temperature: number;
  max_battery_temperature?: number;
  p90_battery_temperature: number;
  avg_battery_discharge: number;
  p90_battery_discharge: number;
  driver_vendor: string;
  driver: string;
  driver_version: string;
  driver_date: string;
  wifi_mac_add: string;
  performance_status: 'active' | 'degraded' | 'critical';
  device_status: 'active' | 'eol_eligible';
}

// ─── API request / response ───────────────────────────────────────────────────

export interface UploadSnapshotRequest {
  snapshot: DeviceSnapshot;
}

export interface UploadSnapshotResponse {
  prediction_id: string;
}

export type PredictionStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface PredictionResponse {
  prediction_id: string;
  status: PredictionStatus;
  root_cause?: string;
  confidence_score?: number;
  slowness_type?: string;
  evidence?: string;
}

// ─── Unified prediction result (same shape for demo + real API) ───────────────

export interface PredictionResult {
  rootCauses: import('@/types/chat.types').RootCause[];
}

// ─── Error ────────────────────────────────────────────────────────────────────

export type ApiErrorCode =
  | 'NETWORK_ERROR'
  | 'TIMEOUT'
  | 'VALIDATION_ERROR'
  | '400' | '401' | '403' | '404' | '422' | '500' | '503';

export interface ApiError {
  code: ApiErrorCode;
  message: string;
  detail?: string;
}
