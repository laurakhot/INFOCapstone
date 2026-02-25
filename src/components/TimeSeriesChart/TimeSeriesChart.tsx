import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, Scatter, ScatterChart, ZAxis, ComposedChart,
  ResponsiveContainer, Legend,
} from 'recharts';
import type { TimeSeriesPoint } from '@/types/chat.types';
import styles from './TimeSeriesChart.module.css';

interface Props {
  points: TimeSeriesPoint[];
  p75Ram: number;
}

function formatDate(dateStr: string) {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

interface ChartPoint extends TimeSeriesPoint {
  restartMarker?: number;
}

export function TimeSeriesChart({ points, p75Ram }: Props) {
  const data: ChartPoint[] = points.map((p) => ({
    ...p,
    // Place restart marker dot at the RAM value for that day
    restartMarker: p.restart ? p.ram_pct : undefined,
  }));

  // Show every 7th date label to avoid clutter
  const tickDates = points
    .filter((_, i) => i % 7 === 0)
    .map((p) => p.date);

  return (
    <div className={styles.container}>
      <div className={styles.heading}>
        📈 30-day RAM &amp; CPU trend
      </div>
      <div className={styles.legend}>
        <span className={styles.legendRam}>— RAM usage</span>
        <span className={styles.legendCpu}>— CPU usage</span>
        <span className={styles.legendP75}>- - P75 benchmark ({p75Ram}%)</span>
        <span className={styles.legendRestart}>● Restart event</span>
      </div>

      <ResponsiveContainer width="100%" height={160}>
        <ComposedChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E8EAEE" />
          <XAxis
            dataKey="date"
            ticks={tickDates}
            tickFormatter={formatDate}
            tick={{ fontSize: 9, fill: '#8F9AA7' }}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fontSize: 9, fill: '#8F9AA7' }}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip
            formatter={(value: number, name: string) => [
              `${value}%`,
              name === 'ram_pct' ? 'RAM' : name === 'cpu_pct' ? 'CPU' : name,
            ]}
            labelFormatter={formatDate}
            contentStyle={{ fontSize: 11, borderColor: '#E8EAEE' }}
          />

          {/* P75 reference line */}
          <ReferenceLine
            y={p75Ram}
            stroke="#8F9AA7"
            strokeDasharray="4 3"
            label={{ value: `P75 ${p75Ram}%`, position: 'right', fontSize: 9, fill: '#8F9AA7' }}
          />

          {/* RAM line */}
          <Line
            type="monotone"
            dataKey="ram_pct"
            stroke="#BA3385"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3 }}
          />

          {/* CPU line */}
          <Line
            type="monotone"
            dataKey="cpu_pct"
            stroke="#8F9AA7"
            strokeWidth={1.5}
            strokeDasharray="3 2"
            dot={false}
            activeDot={{ r: 3 }}
          />

          {/* Restart markers — filled circles on RAM line */}
          <Scatter
            dataKey="restartMarker"
            fill="#EF4444"
            r={4}
            name="Restart"
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className={styles.caption}>
        Pattern: RAM climbs after each restart, always returning above the P75 benchmark —
        confirming sustained insufficiency, not a one-time spike.
      </div>
    </div>
  );
}
