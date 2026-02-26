import type { DeviceSnapshot, LambdaResponse, PredictionResult, ApiError } from '@/types/api.types';
import type { CaseData } from '@/types/chat.types';

const IS_DEMO = (import.meta.env.VITE_DEMO_MODE as string | undefined) !== 'false';
const API_URL = `${(import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '/api'}/predict`;

function delay(ms: number): Promise<void> {
  return new Promise((res) => setTimeout(res, ms));
}

/** Demo mode: return the case JSON's root causes, simulating API latency. */
function localPrediction(caseData: CaseData): PredictionResult {
  return { rootCauses: caseData.rootCauses };
}

/** Real API mode: single POST to the Lambda endpoint, no polling. */
async function apiPrediction(snapshot: DeviceSnapshot): Promise<PredictionResult> {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ snapshot }),
  });

  if (!response.ok) {
    const err: ApiError = {
      code: String(response.status) as ApiError['code'],
      message: `API request failed with status ${response.status}`,
    };
    throw err;
  }

  const data: LambdaResponse = await response.json();
  const { answer, score } = data.prediction;

  return {
    rootCauses: [
      {
        icon: '⚠️',
        label: answer,
        detail: '',
        confidence: Math.round(score * 100),
        type: 'warn',
        evidence: '',
        metrics: [],
      },
    ],
    rawResponse: data,
  };
}

/**
 * Run a prediction for the given device snapshot.
 * In demo mode (VITE_DEMO_MODE !== 'false'), returns the case JSON's mock root causes.
 * In production mode, POSTs to the Lambda endpoint and returns the result.
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
