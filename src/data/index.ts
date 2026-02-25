import type { CaseData } from '@/types/chat.types';
import case1 from './case1.json';
import case2 from './case2.json';
import case3 from './case3.json';

export const CASES: Record<number, CaseData> = {
  1: case1 as unknown as CaseData,
  2: case2 as unknown as CaseData,
  3: case3 as unknown as CaseData,
};

export const CASE_IDS = [1, 2, 3] as const;
export type CaseId = typeof CASE_IDS[number];
