import type { DeviceSnapshot } from '@/types/api.types';

export interface ValidationResult {
  valid: boolean;
  snapshot?: DeviceSnapshot;
  errors: string[];
}

const REQUIRED_FIELDS: (keyof DeviceSnapshot)[] = [
  'agent_id',
  'time',
  'os',
  'os_version',
  'memory_size_gb',
  'cpu_count',
  'uptime_days',
  'p90_memory_utilization',
  'avg_memory_utilization',
  'avg_processor_time',
  'p90_boot_time',
  'avg_boot_time',
  'total_app_crash',
  'device_status',
  'performance_status',
];

/** Validate a JSON file as a device snapshot. Async to read the file contents. */
export async function validateDeviceSnapshotJson(file: File): Promise<ValidationResult> {
  if (!file.name.endsWith('.json') && file.type !== 'application/json') {
    return { valid: false, errors: ['File must be a .json file'] };
  }

  let raw: unknown;
  try {
    const text = await file.text();
    raw = JSON.parse(text);
  } catch {
    return { valid: false, errors: ['File is not valid JSON'] };
  }

  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    return { valid: false, errors: ['Snapshot must be a JSON object'] };
  }

  const obj = raw as Record<string, unknown>;
  const missing = REQUIRED_FIELDS.filter((f) => !(f in obj));

  if (missing.length > 0) {
    return {
      valid: false,
      errors: [`Missing required fields: ${missing.join(', ')}`],
    };
  }

  return { valid: true, snapshot: obj as unknown as DeviceSnapshot, errors: [] };
}
