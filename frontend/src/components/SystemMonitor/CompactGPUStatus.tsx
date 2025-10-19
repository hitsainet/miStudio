/**
 * CompactGPUStatus Component
 *
 * Compact at-a-glance GPU status indicator for navigation bar
 * Shows GPU utilization, VRAM, and temperature
 */

import { Activity, HardDrive, Thermometer } from 'lucide-react';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';
import { MetricValue } from './MetricValue';
import { getTemperatureColor } from '../../utils/metricHelpers';
import { useEffect } from 'react';

export function CompactGPUStatus() {
  const {
    gpuAvailable,
    gpuMetrics,
    isPolling,
    updateInterval,
    startPolling,
  } = useSystemMonitorStore();

  // Start lightweight polling when component mounts
  useEffect(() => {
    if (!isPolling) {
      startPolling(updateInterval || 2000); // Default to 2s for status bar
    }

    return () => {
      // Don't stop polling on unmount - let SystemMonitor page control it
    };
  }, []);

  if (!gpuAvailable || !gpuMetrics) {
    return null;
  }

  const temperature = gpuMetrics.temperature;
  const tempColor = getTemperatureColor(temperature);

  return (
    <div className="flex items-center gap-4 px-4 py-1.5 rounded-lg bg-slate-900/50 border border-slate-700/50 text-xs">
      {/* GPU Utilization */}
      <div className="flex items-center gap-1.5">
        <Activity className="w-3.5 h-3.5 text-emerald-400" />
        <span className="text-slate-400">GPU:</span>
        <span className="text-slate-100 font-medium">
          <MetricValue value={gpuMetrics.utilization?.gpu} format="percent" decimals={0} />
        </span>
      </div>

      {/* VRAM */}
      <div className="flex items-center gap-1.5">
        <HardDrive className="w-3.5 h-3.5 text-blue-400" />
        <span className="text-slate-400">VRAM:</span>
        <span className="text-slate-100 font-medium">
          <MetricValue value={gpuMetrics.memory?.used_gb} format="memory" decimals={1} />
          <span className="text-slate-400">/</span>
          <MetricValue value={gpuMetrics.memory?.total_gb} format="memory" decimals={1} />
        </span>
      </div>

      {/* Temperature */}
      <div className="flex items-center gap-1.5">
        <Thermometer className={`w-3.5 h-3.5 ${tempColor.split(' ')[0]}`} />
        <span className="text-slate-400">Temp:</span>
        <span className={`font-medium ${tempColor.split(' ')[0]}`}>
          <MetricValue value={temperature} format="temperature" decimals={0} />
        </span>
      </div>

      {/* Live indicator */}
      {isPolling && (
        <div className="flex items-center gap-1">
          <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"></div>
        </div>
      )}
    </div>
  );
}
