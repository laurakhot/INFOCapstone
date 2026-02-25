import { CASES } from '@/data/index';
import { useConversation } from '@/state/conversationStore';
import { useChatFlow } from '@/state/useChatFlow';
import { Header } from '@/components/Header/Header';
import { KioskLeft } from '@/components/KioskLeft/KioskLeft';
import { BadgeScreen } from '@/components/BadgeScreen/BadgeScreen';
import { ChatWindow } from '@/components/ChatWindow/ChatWindow';
import styles from './App.module.css';

export default function App() {
  const { state } = useConversation();
  const caseData = CASES[state.currentCase];

  const {
    thinkingState,
    diagnosticState,
    flowDisabled,
    handleBadgeSwipe,
    handleChipClick,
    handleDirectChipClick,
    handleGsdEscape,
    handleSend,
    handleFileSelect,
  } = useChatFlow(caseData);

  return (
    <div className={styles.kiosk}>
      <Header />
      <div className={styles.body}>
        <KioskLeft caseData={caseData} />
        <div className={styles.right}>
          <BadgeScreen onSwipe={handleBadgeSwipe} />
          {state.phase === 'active' && (
            <ChatWindow
              caseData={caseData}
              thinkingState={thinkingState}
              diagnosticState={diagnosticState}
              disabled={flowDisabled}
              onChip={handleChipClick}
              onDirectChip={handleDirectChipClick}
              onGsdEscape={handleGsdEscape}
              onSend={handleSend}
              onFileSelect={handleFileSelect}
            />
          )}
        </div>
      </div>
    </div>
  );
}
