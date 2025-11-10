/**
 * GPUCard Component
 *
 * Compact card showing key metrics for a single GPU in comparison view
 */

import { GPUMetrics, GPUInfo } from '../../types/system';
import { Cpu, Thermometer, Zap, HardDrive } from 'lucide-react';
import { MetricValue } from './MetricValue';
import { getTemperatureColor, getUtilizationColor, safeGet } from '../../utils/metricHelpers';

interface GPUCardProps {
  gpuId: number;
  metrics: GPUMetrics | null;
  info: GPUInfo | null;
}

export function GPUCard({ gpuId, metrics, info }: GPUCardProps) {
  // Safely extract metrics with fallbacks
  const gpuUtil = safeGet(metrics || {}, 'utilization.gpu', 0);
  const memUsedGb = safeGet(metrics || {}, 'memory.used_gb', 0);
  const memTotalGb = safeGet(metrics || {}, 'memory.total_gb', 0);
  const memPercent = safeGet(metrics || {}, 'memory.used_percent', 0);
  const temperature = safeGet(metrics || {}, 'temperature', null);
  const powerUsage = safeGet(metrics || {}, 'power.usage', 0);
  const powerLimit = safeGet(metrics || {}, 'power.limit', 0);
  const powerPercent = safeGet(metrics || {}, 'power.usage_percent', 0);
  const fanSpeed = safeGet(metrics || {}, 'fan_speed', null);

  return (
    <div className="bg-slate-900 rounded-lg border border-slate-800 p-6 space-y-4">
      {/* Header */}
      <div className="border-b border-slate-800 pb-3">
        <div className="flex items-center justify-between mb-1">
          <h3 className="text-lg font-semibold text-slate-100">
            GPU {gpuId}
          </h3>
          <span className={`px-2 py-1 rounded text-xs font-medium ${getTemperatureColor(temperature)}`}>
            <MetricValue value={temperature} format="temperature" decimals={0} />
          </span>
        </div>
        <p className="text-sm text-slate-400 truncate">{info?.name || 'Unknown GPU'}</p>
      </div>

      {/* Metrics Grid */}
      <div className="space-y-3">
        {/* GPU Utilization */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <Cpu className="w-4 h-4" />
              <span>GPU Util</span>
            </div>
            <span className="text-sm font-medium text-slate-100">
              <MetricValue value={gpuUtil} format="percent" />
            </span>
          </div>
          <div className="w-full bg-slate-800 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${getUtilizationColor(gpuUtil)}`}
              style={{ width: `${Math.min(gpuUtil || 0, 100)}%` }}
            ></div>
          </div>
        </div>

        {/* Memory */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <HardDrive className="w-4 h-4" />
              <span>Memory</span>
            </div>
            <span className="text-sm font-medium text-slate-100">
              <MetricValue value={memUsedGb} format="memory" /> / <MetricValue value={memTotalGb} format="memory" />
            </span>
          </div>
          <div className="w-full bg-slate-800 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(memPercent || 0, 100)}%` }}
            ></div>
          </div>
          <div className="text-xs text-slate-500 mt-1">
            <MetricValue value={memPercent} format="percent" />
          </div>
        </div>

        {/* Power */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <Zap className="w-4 h-4" />
              <span>Power</span>
            </div>
            <span className="text-sm font-medium text-slate-100">
              <MetricValue value={powerUsage} format="power" /> / <MetricValue value={powerLimit} format="power" />
            </span>
          </div>
          <div className="w-full bg-slate-800 rounded-full h-2">
            <div
              className="bg-yellow-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(powerPercent || 0, 100)}%` }}
            ></div>
          </div>
          <div className="text-xs text-slate-500 mt-1">
            <MetricValue value={powerPercent} format="percent" />
          </div>
        </div>

        {/* Temperature */}
        <div>
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2 text-slate-400">
              <Thermometer className="w-4 h-4" />
              <span>Temperature</span>
            </div>
            <span className={`font-medium ${getTemperatureColor(temperature).split(' ')[0]}`}>
              <MetricValue value={temperature} format="temperature" decimals={0} />
            </span>
          </div>
        </div>

        {/* Fan Speed */}
        {fanSpeed !== null && fanSpeed !== undefined && fanSpeed > 0 && (
          <div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400">Fan Speed</span>
              <span className="text-slate-100 font-medium">
                <MetricValue value={fanSpeed} format="percent" decimals={0} />
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
