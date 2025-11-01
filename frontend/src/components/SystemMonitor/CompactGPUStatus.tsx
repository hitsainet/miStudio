/**
 * CompactGPUStatus Component
 *
 * Compact at-a-glance system status indicator for navigation bar
 * Shows GPU utilization, VRAM, temperature, CPU, and RAM
 * Clickable to navigate to System Monitor tab
 */

import { Activity, HardDrive, Thermometer, Cpu, MemoryStick } from 'lucide-react';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';
import { MetricValue } from './MetricValue';
import { getTemperatureColor } from '../../utils/metricHelpers';
import { useEffect } from 'react';

interface CompactGPUStatusProps {
  onClickMonitor?: () => void;
}

export function CompactGPUStatus({ onClickMonitor }: CompactGPUStatusProps) {
  const {
    gpuAvailable,
    gpuMetrics,
    systemMetrics,
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

  // Show system metrics even if GPU not available
  const showGPU = gpuAvailable && gpuMetrics;
  const showSystem = systemMetrics;

  // Don't render if no data available at all
  if (!showGPU && !showSystem) {
    return null;
  }

  const temperature = gpuMetrics?.temperature;
  const tempColor = temperature ? getTemperatureColor(temperature) : '';

  return (
    <div
      onClick={onClickMonitor}
      className="flex items-center gap-4 px-4 py-1.5 rounded-lg bg-slate-900/50 border border-slate-700/50 text-xs cursor-pointer hover:bg-slate-800/50 hover:border-slate-600/50 transition-colors"
      title="Click to open System Monitor"
    >
      {/* CPU Utilization */}
      {showSystem && (
        <div className="flex items-center gap-1.5">
          <Cpu className="w-3.5 h-3.5 text-purple-400" />
          <span className="text-slate-400">CPU:</span>
          <span className="text-slate-100 font-medium">
            <MetricValue value={systemMetrics.cpu?.percent} format="percent" decimals={0} />
          </span>
        </div>
      )}

      {/* RAM */}
      {showSystem && (
        <div className="flex items-center gap-1.5">
          <MemoryStick className="w-3.5 h-3.5 text-orange-400" />
          <span className="text-slate-400">RAM:</span>
          <span className="text-slate-100 font-medium">
            <MetricValue value={systemMetrics.ram?.used_gb} format="memory" decimals={1} />
            <span className="text-slate-400">/</span>
            <MetricValue value={systemMetrics.ram?.total_gb} format="memory" decimals={1} />
          </span>
        </div>
      )}

      {/* GPU Utilization */}
      {showGPU && (
        <div className="flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-slate-400">GPU:</span>
          <span className="text-slate-100 font-medium">
            <MetricValue value={gpuMetrics.utilization?.gpu} format="percent" decimals={0} />
          </span>
        </div>
      )}

      {/* VRAM */}
      {showGPU && (
        <div className="flex items-center gap-1.5">
          <HardDrive className="w-3.5 h-3.5 text-blue-400" />
          <span className="text-slate-400">VRAM:</span>
          <span className="text-slate-100 font-medium">
            <MetricValue value={gpuMetrics.memory?.used_gb} format="memory" decimals={1} />
            <span className="text-slate-400">/</span>
            <MetricValue value={gpuMetrics.memory?.total_gb} format="memory" decimals={1} />
          </span>
        </div>
      )}

      {/* Temperature */}
      {showGPU && temperature !== undefined && (
        <div className="flex items-center gap-1.5">
          <Thermometer className={`w-3.5 h-3.5 ${tempColor.split(' ')[0]}`} />
          <span className="text-slate-400">Temp:</span>
          <span className={`font-medium ${tempColor.split(' ')[0]}`}>
            <MetricValue value={temperature} format="temperature" decimals={0} />
          </span>
        </div>
      )}

      {/* Live indicator */}
      {isPolling && (
        <div className="flex items-center gap-1">
          <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"></div>
        </div>
      )}
    </div>
  );
}
