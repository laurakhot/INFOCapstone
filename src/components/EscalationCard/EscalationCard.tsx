import type { EscalationMessage, Chip } from '@/types/chat.types';
import { useConversation } from '@/state/conversationStore';
import styles from './EscalationCard.module.css';

interface Props {
  message: EscalationMessage;
  onChip: (chipId: string) => void;
}

export function EscalationCard({ message, onChip }: Props) {
  const { dispatch } = useConversation();
  const { id, variant, remainingChips, directChips, disabled } = message;

  function markResolved() {
    if (disabled) return;
    dispatch({ type: 'RESOLVE_ESCALATION', messageId: id });
  }

  // ── Self-service: remaining chips or chip-exhausted GSD ──────────────────
  if (variant === 'self_service') {
    return (
      <div className={`${styles.card} ${disabled ? styles.disabled : ''}`}>
        <div className={styles.question}>Does this solve your issue?</div>
        <div className={styles.btnRow}>
          <button className={`${styles.btn} ${styles.yes}`} onClick={markResolved} disabled={disabled}>
            ✅ Yes, issue resolved
          </button>
        </div>
        {remainingChips.length > 0 && (
          <>
            <div className={styles.tryLabel}>If not, try these:</div>
            <div className={styles.chipRow}>
              {remainingChips.map((chip) => (
                <ChipButton key={chip.id} chip={chip} disabled={disabled} onChip={onChip} />
              ))}
            </div>
          </>
        )}
      </div>
    );
  }

  // ── GSD escalation (online agent) ────────────────────────────────────────
  if (variant === 'gsd') {
    return (
      <div className={styles.card}>
        <div className={styles.question}>Connect to a GSD Agent</div>
        <div className={styles.contactBody}>
          <p>📞 <strong>IT Helpdesk:</strong> ext. 1-800</p>
          <p>💬 <strong>Slack:</strong> #it-support</p>
          <p>🌐 <strong>Self-service portal:</strong> my.it.a2z.com</p>
          <p className={styles.note}>Your diagnostic session has been attached to your ticket for context.</p>
        </div>
      </div>
    );
  }

  // ── In-person (hardware failure / upgrade) ────────────────────────────────
  return (
    <div className={styles.card}>
      <div className={styles.question}>Next step — get a replacement laptop</div>
      <div className={styles.directChips}>
        {directChips.map((chip) => (
          <ChipButton key={chip.id} chip={chip} disabled={disabled} onChip={onChip} />
        ))}
      </div>
    </div>
  );
}

function ChipButton({ chip, disabled, onChip }: { chip: Chip; disabled: boolean; onChip: (id: string) => void }) {
  return (
    <button
      className={`${styles.chip} ${chip.primary ? styles.primary : ''}`}
      onClick={() => !disabled && onChip(chip.id)}
      disabled={disabled}
    >
      {chip.label}
    </button>
  );
}
