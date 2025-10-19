/**
 * MemoryUsageChart Component
 *
 * Line chart displaying GPU memory and RAM usage over time
 */

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { MultiSeriesDataPoint } from '../../hooks/useHistoricalData';

interface MemoryUsageChartProps {
  data: MultiSeriesDataPoint[];
}

export function MemoryUsageChart({ data }: MemoryUsageChartProps) {
  // Format timestamp for display
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Format memory in GB
  const formatMemory = (value: number) => {
    return `${value.toFixed(1)} GB`;
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
              {entry.name}: {formatMemory(entry.value)}
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
        Memory Usage Over Time
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
              stroke="#64748b"
              style={{ fontSize: '12px'}}
              tickFormatter={formatMemory}
              label={{ value: 'GB', angle: -90, position: 'insideLeft', style: { fill: '#64748b' } }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: '14px', paddingTop: '10px' }}
              iconType="line"
            />
            <Line
              type="monotone"
              dataKey="gpu_memory_used_gb"
              name="GPU Memory"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="ram_used_gb"
              name="RAM"
              stroke="#6366f1"
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
