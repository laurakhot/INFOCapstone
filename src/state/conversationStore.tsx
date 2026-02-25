import { createContext, useContext, useReducer, type ReactNode } from 'react';
import type {
  ChatMessage,
  ChipsMessage,
  ResolutionMessage,
  EscalationMessage,
} from '@/types/chat.types';
import type { CaseId } from '@/data/index';

// ─── State ────────────────────────────────────────────────────────────────────

export type Phase = 'badge' | 'active';

export interface ConversationState {
  messages: ChatMessage[];
  phase: Phase;
  currentCase: CaseId;
}

// ─── Actions ──────────────────────────────────────────────────────────────────

export type ConversationAction =
  | { type: 'SET_CASE'; caseId: CaseId }
  | { type: 'SET_PHASE'; phase: Phase }
  | { type: 'ADD_MESSAGE'; message: ChatMessage }
  | { type: 'MARK_CHIPS_USED'; messageId: string }
  | { type: 'SET_FEEDBACK'; messageId: string; feedback: 'up' | 'down' }
  | { type: 'RESOLVE_ESCALATION'; messageId: string }
  | { type: 'UPDATE_ESCALATION'; messageId: string; patch: Partial<EscalationMessage> };

// ─── Reducer ──────────────────────────────────────────────────────────────────

const initialState: ConversationState = {
  messages: [],
  phase: 'badge',
  currentCase: 1,
};

function reducer(
  state: ConversationState,
  action: ConversationAction
): ConversationState {
  switch (action.type) {
    case 'SET_CASE':
      // Full reset — returns to badge screen with empty message history
      return { ...initialState, currentCase: action.caseId };

    case 'SET_PHASE':
      return { ...state, phase: action.phase };

    case 'ADD_MESSAGE':
      return { ...state, messages: [...state.messages, action.message] };

    case 'MARK_CHIPS_USED':
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.messageId && m.type === 'chips'
            ? ({ ...(m as ChipsMessage), used: true } as ChipsMessage)
            : m
        ),
      };

    case 'SET_FEEDBACK':
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.messageId && m.type === 'resolution'
            ? ({
                ...(m as ResolutionMessage),
                feedbackGiven: action.feedback,
              } as ResolutionMessage)
            : m
        ),
      };

    case 'RESOLVE_ESCALATION':
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.messageId && m.type === 'escalation'
            ? ({
                ...(m as EscalationMessage),
                resolved: true,
                disabled: true,
              } as EscalationMessage)
            : m
        ),
      };

    case 'UPDATE_ESCALATION':
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.messageId && m.type === 'escalation'
            ? ({ ...(m as EscalationMessage), ...action.patch } as EscalationMessage)
            : m
        ),
      };

    default:
      return state;
  }
}

// ─── Context ──────────────────────────────────────────────────────────────────

interface ConversationContextValue {
  state: ConversationState;
  dispatch: React.Dispatch<ConversationAction>;
}

const ConversationContext = createContext<ConversationContextValue | null>(null);

export function ConversationProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <ConversationContext.Provider value={{ state, dispatch }}>
      {children}
    </ConversationContext.Provider>
  );
}

export function useConversation(): ConversationContextValue {
  const ctx = useContext(ConversationContext);
  if (!ctx) throw new Error('useConversation must be used inside <ConversationProvider>');
  return ctx;
}
