import type { TextMessage } from '@/types/chat.types';
import styles from './MessageBubble.module.css';

interface Props {
  message: TextMessage;
}

export function MessageBubble({ message }: Props) {
  const isAI = message.role === 'ai';

  return (
    <div className={`${styles.msg} ${isAI ? styles.ai : styles.user}`}>
      <div
        className={`${styles.bubble} ${message.small ? styles.small : ''} ${message.isError ? styles.error : ''}`}
        // AI messages may contain pre-sanitised HTML from chatService builders
        dangerouslySetInnerHTML={{ __html: message.html }}
      />
      <div className={styles.time}>{message.timestamp}</div>
    </div>
  );
}
