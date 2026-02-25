import styles from './LoadingIndicator.module.css';

export function LoadingIndicator() {
  return (
    <div className={styles.bubble} aria-label="Husky is typing">
      <div className={styles.dot} />
      <div className={styles.dot} />
      <div className={styles.dot} />
    </div>
  );
}
