import { apiClient } from './apiClient';
import type {
  DeviceSnapshot,
  UploadSnapshotResponse,
  PredictionResponse,
  PredictionResult,
  ApiError,
} from '@/types/api.types';
import type { CaseData } from '@/types/chat.types';

const IS_DEMO = (import.meta.env.VITE_DEMO_MODE as string | undefined) !== 'false';
const POLL_INTERVAL_MS = 2_000;
const MAX_POLLS = 30;

function delay(ms: number): Promise<void> {
  return new Promise((res) => setTimeout(res, ms));
}

/** Demo mode: return the case JSON's root causes, simulating API latency. */
// Returns the root causes already baked into the case JSON file. 
// This is the demo-mode path — no network calls, just extracts 
// caseData.rootCauses and wraps it in the expected PredictionResult shape. 
// Lets the UI render identically whether the backend exists or not.
function localPrediction(caseData: CaseData): PredictionResult {
  return { rootCauses: caseData.rootCauses };
}

/** Real API mode: POST snapshot → poll for result. */
async function apiPrediction(snapshot: DeviceSnapshot): Promise<PredictionResult> {
  const uploadRes = await apiClient.post<UploadSnapshotResponse>('/device-snapshot', {
    snapshot,
  });
  const { prediction_id } = uploadRes.data;

  for (let i = 0; i < MAX_POLLS; i++) {
    await delay(POLL_INTERVAL_MS);
    const pollRes = await apiClient.get<PredictionResponse>(`/prediction/${prediction_id}`);
    const { status, root_cause, confidence_score } = pollRes.data;

    if (status === 'completed' && root_cause) {
      return {
        rootCauses: [
          {
            icon: '⚠️',
            label: root_cause,
            detail: '',
            confidence: confidence_score ?? 0,
            type: 'warn',
            evidence: '',
            metrics: [],
          },
        ],
      };
    }

    if (status === 'failed') {
      const err: ApiError = { code: '500', message: 'Prediction failed on server' };
      throw err;
    }
  }

  const timeoutErr: ApiError = {
    code: 'TIMEOUT',
    message: 'Prediction timed out after 60 seconds',
  };
  throw timeoutErr;
}

/**
 * Run a prediction for the given device snapshot.
 * In demo mode (VITE_DEMO_MODE !== 'false'), returns the case JSON's mock root causes.
 * In production mode, hits the real API and polls for results.
 * Both paths return the same PredictionResult type.
 */
export async function runPrediction(
  snapshot: DeviceSnapshot,
  caseData: CaseData
): Promise<PredictionResult> {
  if (IS_DEMO) {
    await delay(200); // Minimal simulated latency
    return localPrediction(caseData);
  }
  return apiPrediction(snapshot);
}