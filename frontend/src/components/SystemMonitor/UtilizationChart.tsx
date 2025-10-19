/**
 * UtilizationChart Component
 *
 * Line chart displaying GPU and CPU utilization over time
 */

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { MultiSeriesDataPoint } from '../../hooks/useHistoricalData';

interface UtilizationChartProps {
  data: MultiSeriesDataPoint[];
}

export function UtilizationChart({ data }: UtilizationChartProps) {
  // Format timestamp for display
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-3 shadow-lg">
          <p className="text-slate-400 text-xs mb-2">
            {formatTime(payload[0].payload.timestamp)}
          </p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.dataKey === 'gpu_temperature' ? `${entry.value.toFixed(1)}°C` : `${entry.value.toFixed(1)}%`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
      <h3 className="text-lg font-semibold text-slate-100 mb-4">
        Utilization & Temperature
      </h3>
      {data.length === 0 ? (
        <div className="h-64 flex items-center justify-center text-slate-500">
          Collecting data...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data} margin={{ top: 5, right: 40, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatTime}
              stroke="#64748b"
              style={{ fontSize: '12px' }}
            />
            <YAxis
              yAxisId="left"
              domain={[0, 100]}
              stroke="#64748b"
              style={{ fontSize: '12px' }}
              label={{ value: '%', angle: -90, position: 'insideLeft', style: { fill: '#64748b' } }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              domain={[0, 100]}
              stroke="#f59e0b"
              style={{ fontSize: '12px' }}
              label={{ value: '°C', angle: 90, position: 'insideRight', style: { fill: '#f59e0b' } }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: '14px', paddingTop: '10px' }}
              iconType="line"
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="gpu_utilization"
              name="GPU Util"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="cpu_utilization"
              name="CPU Util"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="gpu_temperature"
              name="GPU Temp"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
