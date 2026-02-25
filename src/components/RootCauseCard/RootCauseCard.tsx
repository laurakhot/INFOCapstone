import { useEffect, useRef } from 'react';
import type { CaseData } from '@/types/chat.types';
import styles from './RootCauseCard.module.css';

interface Props {
  caseData: CaseData;
}

export function RootCauseCard({ caseData }: Props) {
  const cardRef = useRef<HTMLDivElement>(null);

  // Animate confidence bars after mount
  useEffect(() => {
    const timeout = setTimeout(() => {
      cardRef.current?.querySelectorAll<HTMLElement>('[data-w]').forEach((el) => {
        el.style.width = el.dataset.w ?? '0%';
      });
    }, 80);
    return () => clearTimeout(timeout);
  }, []);

  const sorted = [...caseData.rootCauses].sort((a, b) => b.confidence - a.confidence);

  return (
    <div className={styles.block} ref={cardRef}>
      <div className={styles.title}>Root cause analysis — ranked by confidence</div>

      <div className={styles.list}>
        {sorted.map((cause, i) => (
          <div key={cause.label} className={`${styles.item} ${cause.type === 'eol' ? styles.eolItem : ''}`}>
            <div className={styles.rank}>#{i + 1}</div>
            <div className={styles.icon}>{cause.icon}</div>
            <div className={styles.info}>
              <div className={styles.label}>
                {cause.label}
                {cause.slownessLabel && (
                  <span className={`${styles.badge} ${styles[cause.slownessType ?? '']}`}>
                    {cause.slownessLabel}
                  </span>
                )}
              </div>
              <div className={styles.detail}>{cause.detail}</div>
              {cause.benchmark && (
                <div className={styles.benchmark}>📊 {cause.benchmark}</div>
              )}
            </div>
            <div className={styles.barWrap}>
              <div className={styles.pct}>{cause.confidence}%</div>
              <div className={styles.bar}>
                <div
                  className={`${styles.fill} ${styles[cause.type]}`}
                  data-w={`${cause.confidence}%`}
                  style={{ width: 0 }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      {caseData.cues.length > 0 && (
        <div className={styles.cueRow}>
          {caseData.cues.map((cue) => (
            <span key={cue} className={styles.cue}>{cue}</span>
          ))}
        </div>
      )}
    </div>
  );
}
