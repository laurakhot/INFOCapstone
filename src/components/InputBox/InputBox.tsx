import { useRef, useState } from 'react';
import styles from './InputBox.module.css';

interface Props {
  onSend: (text: string) => void;
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export function InputBox({ onSend, onFileSelect, disabled }: Props) {
  const [text, setText] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  function handleSend() {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText('');
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
      // Reset so the same file can be re-selected after an error
      e.target.value = '';
    }
  }

  return (
    <div className={styles.bar}>
      {/* Hidden file input — triggered by the attach button */}
      <input
        ref={fileRef}
        type="file"
        accept=".json,application/json"
        className={styles.fileInput}
        onChange={handleFileChange}
        aria-label="Upload device snapshot JSON"
      />

      <button
        className={styles.attachBtn}
        onClick={() => fileRef.current?.click()}
        title="Upload device snapshot (.json)"
        aria-label="Upload device snapshot"
        disabled={disabled}
      >
        📎
      </button>

      <input
        type="text"
        className={styles.textInput}
        placeholder="Type a message…"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        aria-label="Chat message"
      />

      <button
        className={styles.sendBtn}
        onClick={handleSend}
        disabled={!text.trim() || disabled}
        aria-label="Send message"
      >
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <line x1="22" y1="2" x2="11" y2="13" />
          <polygon points="22 2 15 22 11 13 2 9 22 2" />
        </svg>
      </button>
    </div>
  );
}
