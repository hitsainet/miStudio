/**
 * TemperatureChart Component
 *
 * Line chart displaying GPU temperature over time with color-coded zones
 */

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { MultiSeriesDataPoint } from '../../hooks/useHistoricalData';

interface TemperatureChartProps {
  data: MultiSeriesDataPoint[];
  maxTemp?: number;
}

export function TemperatureChart({ data, maxTemp = 95 }: TemperatureChartProps) {
  // Format timestamp for display
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const temp = payload[0].value;
      let status = 'Normal';
      let statusColor = '#10b981';

      if (temp >= 80) {
        status = 'Hot';
        statusColor = '#ef4444';
      } else if (temp >= 70) {
        status = 'Warm';
        statusColor = '#f59e0b';
      }

      return (
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-3 shadow-lg">
          <p className="text-slate-400 text-xs mb-2">
            {formatTime(payload[0].payload.timestamp)}
          </p>
          <p className="text-sm" style={{ color: payload[0].color }}>
            Temperature: {temp.toFixed(1)}°C
          </p>
          <p className="text-xs mt-1" style={{ color: statusColor }}>
            Status: {status}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
      <h3 className="text-lg font-semibold text-slate-100 mb-4">
        Temperature Over Time
      </h3>
      {data.length === 0 ? (
        <div className="h-64 flex items-center justify-center text-slate-500">
          Collecting data...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatTime}
              stroke="#64748b"
              style={{ fontSize: '12px' }}
            />
            <YAxis
              domain={[0, maxTemp + 5]}
              stroke="#64748b"
              style={{ fontSize: '12px' }}
              label={{ value: '°C', angle: -90, position: 'insideLeft', style: { fill: '#64748b' } }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: '14px', paddingTop: '10px' }}
              iconType="line"
            />
            {/* Warning zones */}
            <ReferenceLine y={70} stroke="#f59e0b" strokeDasharray="3 3" opacity={0.5} />
            <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="3 3" opacity={0.5} />
            <Line
              type="monotone"
              dataKey="gpu_temperature"
              name="GPU Temperature"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
      {/* Legend for zones */}
      <div className="flex items-center gap-6 mt-4 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
          <span>Normal (&lt;70°C)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
          <span>Warm (70-80°C)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-500"></div>
          <span>Hot (&gt;80°C)</span>
        </div>
      </div>
    </div>
  );
}
