/**
 * Rank-vs-layer trajectories for the pinned tokens at one position.
 *
 * Three details are requirements rather than styling:
 *  - the Y axis is REVERSED, because rank 1 is the strongest reading;
 *  - the domain is [1, meta.top_n] — not a constant, since the server decides
 *    how deep the readout goes;
 *  - `connectNulls={false}`, because a layer where the token left the top-k is
 *    a gap in the evidence, and bridging it draws a trajectory that was never
 *    measured.
 *
 * Band shading appears only when a BandReport exists (BR-002).
 */

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { PIN_COLORS } from './utils';
import { rankOf } from '../../stores/jlensStore';
import type { BandReport, LensTypeSlice } from '../../types/jlens';

interface TrajectoryChartProps {
  axis: number[];
  slice: LensTypeSlice | undefined;
  pinned: string[];
  topN: number;
  selPos: number;
  bandReport: BandReport | null;
  /**
   * The OTHER lens's slice and axis, drawn dashed beneath the primary.
   *
   * The whole claim of the Jacobian lens is that it sees a token rise EARLIER
   * than the logit lens does. That claim was unfalsifiable from this chart:
   * one lens at a time, so a reader had to switch tabs and remember a shape.
   * Drawn together, the lead is simply visible — or absent, which is equally
   * worth seeing.
   *
   * Its own axis, because a partial artifact and the logit lens do not share
   * one and matching by row index would draw the comparison at wrong layers.
   */
  compareSlice?: LensTypeSlice | undefined;
  compareAxis?: number[];
  compareLabel?: string;
}

export function TrajectoryChart({
  axis,
  slice,
  pinned,
  topN,
  selPos,
  bandReport,
  compareSlice,
  compareAxis,
  compareLabel = 'logit lens',
}: TrajectoryChartProps) {
  /**
   * Series are keyed by an ALIAS, never by the token text.
   *
   * Each chart row is a flat object that also carries the x-axis value under
   * `layer`. Token text goes into the SAME namespace, so pinning the token
   * "layer" — an ordinary English word, and a token in every vocabulary here —
   * overwrites the x-axis value with a rank. The chart then plots every point
   * at x = its own rank, which is wrong in a way that still looks like a chart.
   * Aliasing keeps the two namespaces apart for any token at all; the token
   * text still reaches the tooltip via `name`.
   */
  const series = pinned.map((token, i) => ({ token, key: `s${i}` }));

  // BY ABSOLUTE LAYER on both sides. The two lenses have independent axes, so
  // the comparison series is looked up by layer number rather than row index.
  const compareRow = new Map<number, number>();
  (compareAxis ?? []).forEach((layer, i) => compareRow.set(layer, i));
  const hasCompare = Boolean(compareSlice && (compareAxis ?? []).length);

  const data = axis.map((layer, i) => {
    const row: Record<string, number | null> = { layer };
    for (const { token, key } of series) {
      row[key] = rankOf(slice, i, token);
      if (hasCompare) {
        const ci = compareRow.get(layer);
        row[`c${key}`] =
          ci === undefined ? null : rankOf(compareSlice, ci, token);
      }
    }
    return row;
  });

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
          Rank across layers · position {selPos}
        </span>
        <span className="text-[10px] text-slate-500 dark:text-slate-500">
          lower is stronger · gaps are layers where the token left the top-{topN}
          {hasCompare ? ` · dashed = ${compareLabel}` : ''}
        </span>
      </div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
            <CartesianGrid stroke="#334155" strokeDasharray="2 4" />
            {bandReport && (
              <ReferenceArea
                x1={bandReport.workspace_start}
                x2={bandReport.motor_start}
                fill="#34d399"
                fillOpacity={0.07}
              />
            )}
            <XAxis
              dataKey="layer"
              // NUMERIC, not the default category axis. A band ReferenceArea is
              // positioned by layer NUMBER, and on a category axis those
              // numbers are matched against category labels — which lands the
              // shading in the wrong place, or nowhere, on any sparse axis.
              type="number"
              domain={['dataMin', 'dataMax']}
              allowDecimals={false}
              stroke="#64748b"
              tick={{ fontSize: 10 }}
              label={{
                value: 'layer',
                position: 'insideBottom',
                offset: -2,
                fill: '#64748b',
                fontSize: 10,
              }}
            />
            <YAxis
              reversed
              domain={[1, topN]}
              stroke="#64748b"
              tick={{ fontSize: 10 }}
              allowDecimals={false}
            />
            <Tooltip
              contentStyle={{
                background: '#0f172a',
                border: '1px solid #334155',
                borderRadius: 6,
                fontSize: 11,
              }}
              labelStyle={{ color: '#94a3b8' }}
            />
            {hasCompare &&
              series.map(({ token, key }, i) => (
                <Line
                  key={`c${key}`}
                  type="monotone"
                  dataKey={`c${key}`}
                  name={`${token} · ${compareLabel}`}
                  stroke={PIN_COLORS[i % PIN_COLORS.length]}
                  strokeWidth={1}
                  strokeDasharray="3 3"
                  strokeOpacity={0.55}
                  dot={false}
                  connectNulls={false}
                  isAnimationActive={false}
                />
              ))}
            {series.map(({ token, key }, i) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                name={token}
                stroke={PIN_COLORS[i % PIN_COLORS.length]}
                strokeWidth={2}
                dot={false}
                connectNulls={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
