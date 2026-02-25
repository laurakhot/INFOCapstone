import type { BenchmarkSummary } from '@/types/chat.types';
import styles from './BenchmarkCard.module.css';

interface Props {
  summary: BenchmarkSummary;
}

export function BenchmarkCard({ summary }: Props) {
  return (
    <div className={styles.card}>
      <div className={styles.heading}>
        📊 Your laptop vs. similar {summary.model}s at Amazon
      </div>

      <div className={styles.metrics}>
        {summary.metrics.map((m) => {
          const diff = m.userValue - m.p75Value;
          const diffPct = Math.round((diff / m.p75Value) * 100);
          const isAbove = m.status === 'above';

          return (
            <div key={m.label} className={styles.metricRow}>
              <span className={styles.metricLabel}>{m.label}</span>
              <span className={styles.metricValue}>
                {m.userValue}{m.unit}
              </span>
              <span className={`${styles.metricDiff} ${isAbove ? styles.above : styles.within}`}>
                {isAbove
                  ? `↑ ${diffPct}% higher than 75% of similar devices`
                  : '✓ Within normal range'}
              </span>
            </div>
          );
        })}
      </div>

      <div className={styles.interpretation}>
        💡 {summary.interpretation}
      </div>
    </div>
  );
}
