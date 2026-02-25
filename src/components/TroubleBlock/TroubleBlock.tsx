import type { DiagnosticState } from '@/types/chat.types';
import styles from './TroubleBlock.module.css';

interface Props {
  state: DiagnosticState;
}

export function TroubleBlock({ state }: Props) {
  return (
    <div className={`${styles.block} ${state.fading ? styles.fading : ''}`}>
      <div className={styles.title}>📊 Diagnostic results</div>
      {state.rows.map((row, i) => (
        <div
          key={i}
          className={`${styles.row} ${row.visible ? styles.visible : ''}`}
        >
          <span className={styles.label}>{row.label}</span>
          <span className={`${styles.result} ${styles[row.cls]}`}>{row.result}</span>
        </div>
      ))}
    </div>
  );
}
