import { useConversation } from '@/state/conversationStore';
import { CASE_IDS, type CaseId } from '@/data/index';
import styles from './Header.module.css';

const CASE_LABELS: Record<number, string> = {
  1: 'Case 1 · macOS Slow',
  2: 'Case 2 · Hardware Failure',
  3: 'Case 3 · Insufficient RAM',
};

export function Header() {
  const { state, dispatch } = useConversation();

  function handleCaseClick(caseId: CaseId) {
    if (caseId !== state.currentCase) {
      dispatch({ type: 'SET_CASE', caseId });
    }
  }

  return (
    <header className={styles.header}>
      <div className={styles.brand}>
        <div className={styles.brandIcon}>🐾</div>
        <div>
          <div className={styles.brandName}>Husky IT Support</div>
          <div className={styles.brandSub}>Kiosk Concept Demo</div>
        </div>
      </div>

      <div className={styles.controls}>
        <span className={styles.ctrlLabel}>Scenario</span>
        <div className={styles.pillGroup} role="group" aria-label="Scenario selector">
          {CASE_IDS.map((id) => (
            <button
              key={id}
              className={`${styles.pillBtn} ${state.currentCase === id ? styles.active : ''}`}
              onClick={() => handleCaseClick(id)}
              aria-pressed={state.currentCase === id}
            >
              {CASE_LABELS[id]}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
}
