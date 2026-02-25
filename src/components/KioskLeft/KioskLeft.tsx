import type { CaseData } from '@/types/chat.types';
import { useConversation } from '@/state/conversationStore';
import styles from './KioskLeft.module.css';

interface Props {
  caseData: CaseData;
}

export function KioskLeft({ caseData }: Props) {
  const { state } = useConversation();
  const isActive = state.phase !== 'badge';

  return (
    <aside className={styles.panel}>
      {/* Decorative circles */}
      <div className={styles.circle} style={{ width: 180, height: 180, top: -60, right: -60 }} />
      <div className={styles.circle} style={{ width: 120, height: 120, bottom: 40, left: -40 }} />

      <div className={styles.logo}>🐾</div>
      <div className={styles.name}>Husky</div>
      <div className={styles.sub}>AI IT Assistant</div>
      <div className={styles.divider} />

      <div className={styles.context}>
        {isActive ? <ActiveContext caseData={caseData} /> : <IdleHint />}
      </div>

      <div className={styles.footer}>IT Support Kiosk</div>
    </aside>
  );
}

function IdleHint() {
  return <p className={styles.hint}>Swipe your badge<br />to get started</p>;
}

function ActiveContext({ caseData }: { caseData: CaseData }) {
  const h = caseData.deviceHealth;

  return (
    <>
      <div className={styles.userRow}>
        <div className={styles.avatar}>👤</div>
        <div>
          <div className={styles.userName}>{caseData.user}</div>
        </div>
      </div>

      <div className={styles.sectionLabel}>Device</div>
      <div className={styles.stat}>
        <span className={styles.statIcon}>💻</span>
        <span className={styles.statVal}>{caseData.device}</span>
      </div>

      <div className={styles.sectionLabel}>Health Signals</div>
      <div className={styles.signals}>
        {h.signals.map((sig) => (
          <div key={sig.label} className={styles.signal}>
            <span className={styles.signalLabel}>{sig.icon} {sig.label}</span>
            <span className={styles.signalVal}>{sig.value}</span>
          </div>
        ))}
      </div>
    </>
  );
}
