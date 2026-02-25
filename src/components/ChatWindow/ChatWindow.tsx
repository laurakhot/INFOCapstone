import { useEffect, useRef } from 'react';
import type {
  CaseData,
  ThinkingState,
  DiagnosticState,
  ChatMessage,
} from '@/types/chat.types';
import { useConversation } from '@/state/conversationStore';
import { MessageBubble } from '@/components/MessageBubble/MessageBubble';
import { LoadingIndicator } from '@/components/LoadingIndicator/LoadingIndicator';
import { ChipRow } from '@/components/ChipRow/ChipRow';
import { ThinkingBlock } from '@/components/ThinkingBlock/ThinkingBlock';
import { TroubleBlock } from '@/components/TroubleBlock/TroubleBlock';
import { RootCauseCard } from '@/components/RootCauseCard/RootCauseCard';
import { BenchmarkCard } from '@/components/BenchmarkCard/BenchmarkCard';
import { TimeSeriesChart } from '@/components/TimeSeriesChart/TimeSeriesChart';
import { ResolutionCard } from '@/components/ResolutionCard/ResolutionCard';
import { EscalationCard } from '@/components/EscalationCard/EscalationCard';
import { InputBox } from '@/components/InputBox/InputBox';
import styles from './ChatWindow.module.css';

interface Props {
  caseData: CaseData;
  thinkingState: ThinkingState | null;
  diagnosticState: DiagnosticState | null;
  disabled: boolean;
  onChip: (chipId: string) => void;
  onDirectChip: (chipId: string) => void;
  onGsdEscape: () => void;
  onSend: (text: string) => void;
  onFileSelect: (file: File) => void;
}

export function ChatWindow({
  caseData,
  thinkingState,
  diagnosticState,
  disabled,
  onChip,
  onDirectChip,
  onGsdEscape,
  onSend,
  onFileSelect,
}: Props) {
  const { state, dispatch } = useConversation();
  const { messages, phase } = state;
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom whenever messages or ephemeral blocks change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, thinkingState, diagnosticState]);

  if (phase === 'badge') return null;

  const isAnimating = thinkingState !== null || diagnosticState !== null;
  const showLoading = disabled && !isAnimating;
  const showGsdBtn =
    caseData.resolution === 'self_service' &&
    !messages.some((m) => m.type === 'escalation' && m.variant === 'gsd');

  return (
    <div className={styles.window}>
      {/* Message list */}
      <div className={styles.messages}>
        {messages.map((msg) => renderMessage(msg, dispatch, onChip, onDirectChip))}

        {thinkingState && <ThinkingBlock state={thinkingState} />}
        {diagnosticState && <TroubleBlock state={diagnosticState} />}
        {showLoading && <LoadingIndicator />}

        <div ref={bottomRef} />
      </div>

      {/* GSD escape hatch — always visible during self-service flow */}
      {showGsdBtn && (
        <div className={styles.escapeRow}>
          <button
            className={styles.escapeBtn}
            onClick={onGsdEscape}
            disabled={disabled}
          >
            💬 Talk to a GSD agent
          </button>
        </div>
      )}

      {/* Text input */}
      <InputBox onSend={onSend} onFileSelect={onFileSelect} disabled={disabled} />
    </div>
  );
}

// ─── Message renderer ─────────────────────────────────────────────────────────

function renderMessage(
  msg: ChatMessage,
  dispatch: ReturnType<typeof useConversation>['dispatch'],
  onChip: (id: string) => void,
  onDirectChip: (id: string) => void
): React.ReactNode {
  switch (msg.type) {
    case 'text':
      return <MessageBubble key={msg.id} message={msg} />;

    case 'chips':
      return (
        <ChipRow
          key={msg.id}
          chips={msg.chips}
          used={msg.used}
          onChip={(chipId) => {
            if (!msg.used) {
              dispatch({ type: 'MARK_CHIPS_USED', messageId: msg.id });
              onChip(chipId);
            }
          }}
        />
      );

    case 'root-cause':
      return (
        <div key={msg.id} className="root-cause-group">
          <RootCauseCard caseData={msg.caseData} />
          <BenchmarkCard summary={msg.caseData.benchmarkSummary} />
        </div>
      );

    case 'resolution':
      return <ResolutionCard key={msg.id} message={msg} />;

    case 'escalation':
      return (
        <EscalationCard
          key={msg.id}
          message={msg}
          onChip={(chipId) => {
            if (msg.variant === 'in_person') {
              onDirectChip(chipId);
            } else {
              onChip(chipId);
            }
          }}
        />
      );

    case 'timeseries':
      return (
        <TimeSeriesChart
          key={msg.id}
          points={msg.points}
          p75Ram={msg.p75Ram}
        />
      );

    default:
      return null;
  }
}
