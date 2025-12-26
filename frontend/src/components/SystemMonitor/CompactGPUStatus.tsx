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
    gpuList,
    allGpuMetrics,
    systemMetrics,
    isPolling,
    updateInterval,
    startPolling,
    fetchAllGpuMetrics,
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

  // Also fetch all GPU metrics for multi-GPU display
  useEffect(() => {
    if (gpuAvailable && gpuList && gpuList.gpus.length > 1) {
      fetchAllGpuMetrics();
      const intervalId = setInterval(() => {
        fetchAllGpuMetrics();
      }, updateInterval || 2000);
      return () => clearInterval(intervalId);
    }
    return undefined;
  }, [gpuAvailable, gpuList, fetchAllGpuMetrics, updateInterval]);

  // Show system metrics even if GPU not available
  const showGPU = gpuAvailable && gpuMetrics;
  const showSystem = systemMetrics;
  const hasMultipleGPUs = gpuList && gpuList.gpus.length > 1;

  // Don't render if no data available at all
  if (!showGPU && !showSystem) {
    return null;
  }

  // For multi-GPU: calculate combined VRAM and get max temperature
  let totalVramUsed = 0;
  let totalVramTotal = 0;
  let maxTemp = 0;

  if (hasMultipleGPUs && Object.keys(allGpuMetrics).length > 0) {
    Object.values(allGpuMetrics).forEach((metrics: any) => {
      if (metrics?.memory) {
        totalVramUsed += metrics.memory.used_gb || 0;
        totalVramTotal += metrics.memory.total_gb || 0;
      }
      if (metrics?.temperature && metrics.temperature > maxTemp) {
        maxTemp = metrics.temperature;
      }
    });
  } else if (gpuMetrics) {
    totalVramUsed = gpuMetrics.memory?.used_gb || 0;
    totalVramTotal = gpuMetrics.memory?.total_gb || 0;
    maxTemp = gpuMetrics.temperature || 0;
  }

  const tempColor = maxTemp ? getTemperatureColor(maxTemp) : '';

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

      {/* GPU Utilization - Show all GPUs */}
      {showGPU && hasMultipleGPUs && Object.keys(allGpuMetrics).length > 0 ? (
        <div className="flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-slate-400">GPU:</span>
          <span className="text-slate-100 font-medium flex items-center gap-1">
            {gpuList?.gpus.map((gpu, idx) => {
              const metrics = allGpuMetrics[gpu.gpu_id];
              const util = metrics?.utilization?.gpu || 0;
              return (
                <span key={gpu.gpu_id} className="flex items-center">
                  {idx > 0 && <span className="text-slate-500 mx-0.5">|</span>}
                  <span className={util > 80 ? 'text-yellow-400' : ''}>
                    <MetricValue value={util} format="percent" decimals={0} />
                  </span>
                </span>
              );
            })}
          </span>
        </div>
      ) : showGPU && (
        <div className="flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-slate-400">GPU:</span>
          <span className="text-slate-100 font-medium">
            <MetricValue value={gpuMetrics.utilization?.gpu} format="percent" decimals={0} />
          </span>
        </div>
      )}

      {/* VRAM - Combined for multi-GPU */}
      {showGPU && (
        <div className="flex items-center gap-1.5">
          <HardDrive className="w-3.5 h-3.5 text-blue-400" />
          <span className="text-slate-400">VRAM:</span>
          <span className="text-slate-100 font-medium">
            <MetricValue value={totalVramUsed} format="memory" decimals={1} />
            <span className="text-slate-400">/</span>
            <MetricValue value={totalVramTotal} format="memory" decimals={1} />
          </span>
        </div>
      )}

      {/* Temperature - Max across GPUs */}
      {showGPU && maxTemp > 0 && (
        <div className="flex items-center gap-1.5">
          <Thermometer className={`w-3.5 h-3.5 ${tempColor.split(' ')[0]}`} />
          <span className="text-slate-400">Temp:</span>
          <span className={`font-medium ${tempColor.split(' ')[0]}`}>
            <MetricValue value={maxTemp} format="temperature" decimals={0} />
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
