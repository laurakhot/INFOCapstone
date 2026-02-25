import axios from 'axios';
import type { ApiError } from '@/types/api.types';

const BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '/api';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,
  headers: { 'Content-Type': 'application/json' },
});

// Normalize all errors to ApiError shape
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    let apiError: ApiError;

    if (axios.isAxiosError(error)) {
      if (!error.response) {
        apiError = {
          code: 'NETWORK_ERROR',
          message: 'Network error — check your connection',
        };
      } else {
        const status = String(error.response.status) as ApiError['code'];
        const detail =
          typeof (error.response.data as Record<string, unknown>)?.detail === 'string'
            ? String((error.response.data as Record<string, unknown>).detail)
            : undefined;
        apiError = { code: status, message: error.message, detail };
      }
    } else {
      apiError = { code: 'NETWORK_ERROR', message: String(error) };
    }

    return Promise.reject(apiError);
  }
);
