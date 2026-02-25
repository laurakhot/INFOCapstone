import type {
  ChatMessage,
  TextMessage,
  ChipsMessage,
  RootCauseMessage,
  ResolutionMessage,
  EscalationMessage,
  TimeSeriesMessage,
  Chip,
  Resolution,
  CaseData,
  TimeSeriesPoint,
  EscalationVariant,
  MessageRole,
} from '@/types/chat.types';

let _counter = 0;

function newId(): string {
  return `msg-${Date.now()}-${++_counter}`;
}

// workhorse — used for greetings, explanations, error messages, 
// and any freeform text in the conversation.
export function buildTextMessage(
  html: string,
  role: MessageRole = 'ai',
  opts?: { small?: boolean; isError?: boolean }
): TextMessage {
  return {
    id: newId(),
    type: 'text',
    role,
    html,
    timestamp: new Date().toISOString(),
    ...opts,
  };
}

export function buildChipsMessage(chips: Chip[]): ChipsMessage {
  return { id: newId(), type: 'chips', chips, used: false };
}

export function buildRootCauseMessage(caseData: CaseData): RootCauseMessage {
  return { id: newId(), type: 'root-cause', caseData };
}

export function buildResolutionMessage(resolution: Resolution): ResolutionMessage {
  return { id: newId(), type: 'resolution', resolution };
}

export function buildEscalationMessage(
  variant: EscalationVariant,
  remainingChips: Chip[],
  directChips: Chip[]
): EscalationMessage {
  return {
    id: newId(),
    type: 'escalation',
    variant,
    remainingChips,
    directChips,
    resolved: false,
    disabled: false,
  };
}

export function buildTimeSeriesMessage(
  points: TimeSeriesPoint[],
  p75Ram: number
): TimeSeriesMessage {
  return { id: newId(), type: 'timeseries', points, p75Ram };
}

// Re-export the union type for convenience
export type { ChatMessage };