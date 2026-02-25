import type { Chip } from '@/types/chat.types';
import styles from './ChipRow.module.css';

interface Props {
  chips: Chip[];
  used: boolean;
  onChip: (chipId: string) => void;
}

export function ChipRow({ chips, used, onChip }: Props) {
  return (
    <div className={styles.row}>
      {chips.map((chip) => (
        <button
          key={chip.id}
          className={`${styles.chip} ${chip.primary ? styles.primary : ''} ${used ? styles.used : ''}`}
          onClick={() => !used && onChip(chip.id)}
          disabled={used}
          aria-disabled={used}
        >
          {chip.label}
        </button>
      ))}
    </div>
  );
}
