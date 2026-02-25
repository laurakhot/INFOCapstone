export function formatPercent(value: number, decimals = 1): string {
  return `${value.toFixed(decimals)}%`;
}

export function formatSeconds(value: number): string {
  return `${value}s`;
}

export function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function formatBenchmarkDiff(
  userValue: number,
  p75Value: number,
  unit: string
): string {
  const diff = userValue - p75Value;
  const pct = Math.round((Math.abs(diff) / p75Value) * 100);
  if (diff > 0) {
    return `↑ ${pct}% higher than 75% of similar devices (P75: ${p75Value}${unit})`;
  }
  return `✓ Within normal range (P75: ${p75Value}${unit})`;
}

export function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
