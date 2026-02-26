// ─── Enums / primitives ───────────────────────────────────────────────────────

export type ResolutionPath = 'self_service' | 'it_support_desk' | 'hardware_upgrade';
export type UseCase = 'resource_optimization' | 'hardware_failure' | 'hardware_insufficiency';
export type SlownessType =
  | 'memory-leak'
  | 'thrashing'
  | 'disk-bottleneck'
  | 'capacity-mismatch'
  | 'high-utilization'
  | 'hardware-degradation';

// ─── Root cause ───────────────────────────────────────────────────────────────

export interface RootCause {
  icon: string;
  label: string;
  detail: string;
  confidence: number;
  /** 'eol' drives blue styling; 'warn' drives amber. */
  type: 'warn' | 'eol';
  /** Only present when the root cause is a slowness/performance issue. */
  slownessType?: SlownessType;
  slownessLabel?: string;
  evidence: string;
  metrics: string[];
  /** P75 comparison sentence shown below the confidence bar. */
  benchmark?: string;
}

// ─── Diagnostic row ──────────────────────────────────────────────────────────
// Unified type for the ephemeral "📊 Diagnostic results" table.
// Rows can represent raw metric readings ("Device uptime: ⚠️ 108 days") and/or
// prediction-logic steps ("uptime > 60 days → Memory pressure likely ⚠️").
// This eliminates a separate PredictionLogStep type and a second ephemeral block.

export interface DiagnosticRow {
  label: string;
  result: string;
  cls: 'warn' | 'ok' | 'eol';
}

// ─── Chips / Responses ────────────────────────────────────────────────────────

export interface Chip {
  id: string;
  label: string;
  primary?: boolean;
}

export interface Resolution {
  icon: string;
  title: string;
  body: string;
  source: string;
}

// ─── Benchmark card (P75 comparison) ─────────────────────────────────────────

export interface BenchmarkMetric {
  label: string;
  userValue: number;
  p75Value: number;
  unit: string;
  status: 'above' | 'within';
}

export interface BenchmarkSummary {
  model: string;
  metrics: BenchmarkMetric[];
  interpretation: string;
}

// ─── Time series (Case 3 only) ────────────────────────────────────────────────

export interface TimeSeriesPoint {
  date: string;
  ram_pct: number;
  cpu_pct: number;
  restart: boolean;
}

// ─── Case data (loaded from /data/case*.json) ─────────────────────────────────

export interface CaseData {
  id: string;
  label: string;
  useCase: UseCase;
  resolution: ResolutionPath;
  user: string;
  greeting: string;
  device: string;
  snapshot: Record<string, unknown>;
  thinkingSteps: string[];
  /** Unified diagnostic + prediction-logic rows shown in one ephemeral table. */
  diagnosticRows: DiagnosticRow[];
  rootCauses: RootCause[];
  deviceHealth: DeviceHealth;
  benchmarkSummary: BenchmarkSummary;
  cues: string[];
  chips: Chip[];
  /** Chips shown on the in-person escalation card (hardware paths). */
  directEscalationChips?: Chip[];
  responses: Record<string, Resolution>;
  timeSeries?: TimeSeriesPoint[];
}

// ─── Chat message union ────────────────────────────────────────────────────────

export type MessageRole = 'ai' | 'user';

export interface TextMessage {
  id: string;
  type: 'text';
  role: MessageRole;
  html: string;
  timestamp: string;
  small?: boolean;
  isError?: boolean;
}

export interface ChipsMessage {
  id: string;
  type: 'chips';
  chips: Chip[];
  used: boolean;
}

export interface RootCauseMessage {
  id: string;
  type: 'root-cause';
  caseData: CaseData;
  /** Raw Lambda response — only set when VITE_DEMO_MODE=false, for testing. */
  apiDebug?: {
    answer: string;
    score: number;
  };
}

export interface ResolutionMessage {
  id: string;
  type: 'resolution';
  resolution: Resolution;
  feedbackGiven?: 'up' | 'down';
}

export type EscalationVariant = 'self_service' | 'gsd' | 'in_person';

export interface EscalationMessage {
  id: string;
  type: 'escalation';
  variant: EscalationVariant;
  /** Non-empty when variant === 'self_service' and chips remain. */
  remainingChips: Chip[];
  /** Non-empty when variant === 'in_person'. */
  directChips: Chip[];
  resolved: boolean;
  disabled: boolean;
}

export interface TimeSeriesMessage {
  id: string;
  type: 'timeseries';
  points: TimeSeriesPoint[];
  /** P75 RAM value drawn as a reference line. */
  p75Ram: number;
}

export type ChatMessage =
  | TextMessage
  | ChipsMessage
  | RootCauseMessage
  | ResolutionMessage
  | EscalationMessage
  | TimeSeriesMessage;

// ─── Ephemeral animation state (owned by useChatFlow, never stored) ───────────

export interface ThinkingStepItem {
  text: string;
  visible: boolean;
  done: boolean;
}

export interface ThinkingState {
  steps: ThinkingStepItem[];
  fading: boolean;
}

export interface DiagnosticRowItem extends DiagnosticRow {
  visible: boolean;
}

export interface DiagnosticState {
  rows: DiagnosticRowItem[];
  fading: boolean;
}