import { useConversation } from '@/state/conversationStore';
import styles from './BadgeScreen.module.css';

interface Props {
  onSwipe: () => void;
}

export function BadgeScreen({ onSwipe }: Props) {
  const { state } = useConversation();
  const hidden = state.phase !== 'badge';

  function handleClick() {
    if (!hidden) onSwipe();
  }

  return (
    <div
      className={`${styles.screen} ${hidden ? styles.hidden : ''}`}
      onClick={handleClick}
      role="button"
      tabIndex={hidden ? -1 : 0}
      aria-label="Click to simulate badge swipe"
      onKeyDown={(e) => e.key === 'Enter' && handleClick()}
    >
      <div className={styles.logo}>🐾</div>
      <div className={styles.title}>IT Support Kiosk</div>
      <div className={styles.subtitle}>Husky AI Assistant</div>
      <div className={styles.card}>
        <div className={styles.cardIcon}>💳</div>
        <div className={styles.cardText}>Swipe your badge</div>
      </div>
      <div className={styles.hint}>Click anywhere to simulate badge swipe</div>
    </div>
  );
}
