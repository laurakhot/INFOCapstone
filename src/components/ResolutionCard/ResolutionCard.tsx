import type { ResolutionMessage } from '@/types/chat.types';
import { useConversation } from '@/state/conversationStore';
import styles from './ResolutionCard.module.css';

interface Props {
  message: ResolutionMessage;
}

export function ResolutionCard({ message }: Props) {
  const { dispatch } = useConversation();
  const { resolution, feedbackGiven, id } = message;
  const hasFeedback = feedbackGiven != null;

  function handleFeedback(vote: 'up' | 'down') {
    if (hasFeedback) return;
    dispatch({ type: 'SET_FEEDBACK', messageId: id, feedback: vote });
  }

  return (
    <div className={styles.card}>
      <div className={styles.title}>{resolution.icon} {resolution.title}</div>
      <div
        className={styles.body}
        dangerouslySetInnerHTML={{ __html: resolution.body }}
      />
      <div className={styles.source}>📎 {resolution.source}</div>

      <div className={styles.feedbackRow}>
        <span className={styles.feedbackLabel}>Helpful?</span>
        <button
          className={`${styles.fbBtn} ${feedbackGiven === 'up' ? styles.up : ''}`}
          onClick={() => handleFeedback('up')}
          disabled={hasFeedback}
          aria-label="Helpful"
        >
          👍
        </button>
        <button
          className={`${styles.fbBtn} ${feedbackGiven === 'down' ? styles.down : ''}`}
          onClick={() => handleFeedback('down')}
          disabled={hasFeedback}
          aria-label="Not helpful"
        >
          👎
        </button>
      </div>
    </div>
  );
}
