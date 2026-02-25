import type { ThinkingState } from '@/types/chat.types';
import styles from './ThinkingBlock.module.css';

interface Props {
  state: ThinkingState;
}

export function ThinkingBlock({ state }: Props) {
  return (
    <div className={`${styles.block} ${state.fading ? styles.fading : ''}`}>
      <div className={styles.title}>🔍 Analyzing your device</div>
      {state.steps.map((step, i) => (
        <div
          key={i}
          className={`${styles.step} ${step.visible ? styles.visible : ''} ${step.done ? styles.done : ''}`}
        >
          {step.done
            ? <span className={styles.check}>✓</span>
            : <span className={styles.spinner} aria-hidden="true" />
          }
          <span>{step.text}</span>
        </div>
      ))}
    </div>
  );
}
