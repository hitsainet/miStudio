/**
 * System Monitor Page
 *
 * Main page component for system and GPU monitoring dashboard.
 * Displays real-time metrics with auto-refresh.
 */

import { useEffect } from 'react';
import { Activity } from 'lucide-react';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';

export function SystemMonitor() {
  const {
    gpuAvailable,
    gpuMetrics,
    gpuInfo,
    gpuProcesses,
    systemMetrics,
    diskUsage,
    loading,
    error,
    isPolling,
    startPolling,
    stopPolling,
  } = useSystemMonitorStore();

  // Start polling on mount, stop on unmount
  useEffect(() => {
    startPolling(1000); // Poll every 1 second

    return () => {
      stopPolling();
    };
  }, [startPolling, stopPolling]);

  if (loading && !systemMetrics) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-slate-400">Loading system metrics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-red-400">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950">
      <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="w-8 h-8 text-emerald-400" />
          <div>
            <h1 className="text-2xl font-bold text-slate-100">System Monitor</h1>
            <p className="text-sm text-slate-400">
              Real-time GPU and system resource monitoring
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isPolling && (
            <div className="flex items-center gap-2 text-sm text-emerald-400">
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
              <span>Live</span>
            </div>
          )}
        </div>
      </div>

      {/* GPU Section */}
      {gpuAvailable && gpuMetrics && gpuInfo ? (
        <div className="space-y-4">
          {/* GPU Header */}
          <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
            <h2 className="text-lg font-semibold text-slate-100 mb-2">GPU Information</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-slate-400">Device</div>
                <div className="text-slate-100 font-medium">{gpuInfo.name}</div>
              </div>
              <div>
                <div className="text-slate-400">Driver</div>
                <div className="text-slate-100 font-medium">{gpuInfo.driver_version}</div>
              </div>
              <div>
                <div className="text-slate-400">CUDA</div>
                <div className="text-slate-100 font-medium">{gpuInfo.cuda_version}</div>
              </div>
              <div>
                <div className="text-slate-400">Memory</div>
                <div className="text-slate-100 font-medium">{gpuInfo.total_memory_gb} GB</div>
              </div>
            </div>
          </div>

          {/* GPU Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* GPU Utilization */}
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
              <div className="text-sm text-slate-400 mb-2">GPU Utilization</div>
              <div className="text-3xl font-bold text-slate-100 mb-3">
                {gpuMetrics.utilization.gpu.toFixed(1)}%
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div
                  className="bg-emerald-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(gpuMetrics.utilization.gpu, 100)}%` }}
                ></div>
              </div>
            </div>

            {/* GPU Memory */}
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
              <div className="text-sm text-slate-400 mb-2">GPU Memory</div>
              <div className="text-3xl font-bold text-slate-100 mb-1">
                {gpuMetrics.memory.used_percent.toFixed(1)}%
              </div>
              <div className="text-xs text-slate-400 mb-2">
                {gpuMetrics.memory.used_gb.toFixed(2)} / {gpuMetrics.memory.total_gb.toFixed(2)} GB
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(gpuMetrics.memory.used_percent, 100)}%` }}
                ></div>
              </div>
            </div>

            {/* GPU Temperature */}
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
              <div className="text-sm text-slate-400 mb-2">Temperature</div>
              <div className="text-3xl font-bold text-slate-100 mb-3">
                {gpuMetrics.temperature}°C
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-300 ${
                    gpuMetrics.temperature > 80
                      ? 'bg-red-500'
                      : gpuMetrics.temperature > 60
                      ? 'bg-yellow-500'
                      : 'bg-green-500'
                  }`}
                  style={{ width: `${Math.min((gpuMetrics.temperature / 100) * 100, 100)}%` }}
                ></div>
              </div>
            </div>

            {/* GPU Power */}
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
              <div className="text-sm text-slate-400 mb-2">Power Usage</div>
              <div className="text-3xl font-bold text-slate-100 mb-1">
                {gpuMetrics.power.usage_percent.toFixed(1)}%
              </div>
              <div className="text-xs text-slate-400 mb-2">
                {gpuMetrics.power.usage.toFixed(1)} / {gpuMetrics.power.limit.toFixed(0)} W
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div
                  className="bg-yellow-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(gpuMetrics.power.usage_percent, 100)}%` }}
                ></div>
              </div>
            </div>
          </div>

          {/* GPU Processes */}
          {gpuProcesses && gpuProcesses.length > 0 && (
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
              <h3 className="text-lg font-semibold text-slate-100 mb-3">
                GPU Processes ({gpuProcesses.length} active)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-slate-400 border-b border-slate-800">
                    <tr>
                      <th className="text-left py-2 px-2">PID</th>
                      <th className="text-left py-2 px-2">Process</th>
                      <th className="text-right py-2 px-2">GPU Memory</th>
                    </tr>
                  </thead>
                  <tbody className="text-slate-100">
                    {gpuProcesses.slice(0, 10).map((proc) => (
                      <tr key={proc.pid} className="border-b border-slate-800/50">
                        <td className="py-2 px-2">{proc.pid}</td>
                        <td className="py-2 px-2 font-mono text-xs">{proc.process_name}</td>
                        <td className="py-2 px-2 text-right">{proc.gpu_memory_used_mb.toFixed(0)} MB</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
          <div className="text-slate-400 text-center">
            GPU monitoring not available. Ensure NVIDIA GPU and drivers are installed.
          </div>
        </div>
      )}

      {/* System Metrics Section */}
      {systemMetrics && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-slate-100">System Resources</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* CPU */}
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
              <div className="text-sm text-slate-400 mb-2">CPU Utilization</div>
              <div className="text-3xl font-bold text-slate-100 mb-1">
                {systemMetrics.cpu.percent.toFixed(1)}%
              </div>
              <div className="text-xs text-slate-400 mb-2">{systemMetrics.cpu.count} cores</div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div
                  className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(systemMetrics.cpu.percent, 100)}%` }}
                ></div>
              </div>
            </div>

            {/* RAM */}
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
              <div className="text-sm text-slate-400 mb-2">RAM Usage</div>
              <div className="text-3xl font-bold text-slate-100 mb-1">
                {systemMetrics.ram.used_percent.toFixed(1)}%
              </div>
              <div className="text-xs text-slate-400 mb-2">
                {systemMetrics.ram.used_gb.toFixed(1)} / {systemMetrics.ram.total_gb.toFixed(1)} GB
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div
                  className="bg-cyan-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(systemMetrics.ram.used_percent, 100)}%` }}
                ></div>
              </div>
            </div>

            {/* Swap */}
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
              <div className="text-sm text-slate-400 mb-2">Swap Usage</div>
              <div className="text-3xl font-bold text-slate-100 mb-1">
                {systemMetrics.swap.used_percent.toFixed(1)}%
              </div>
              <div className="text-xs text-slate-400 mb-2">
                {systemMetrics.swap.used_gb.toFixed(1)} / {systemMetrics.swap.total_gb.toFixed(1)} GB
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div
                  className="bg-orange-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(systemMetrics.swap.used_percent, 100)}%` }}
                ></div>
              </div>
            </div>

            {/* Disk I/O */}
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
              <div className="text-sm text-slate-400 mb-2">Disk I/O</div>
              <div className="text-sm text-slate-100">
                <div className="flex justify-between mb-1">
                  <span className="text-slate-400">Read:</span>
                  <span>{systemMetrics.disk_io.read_mb.toFixed(0)} MB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Write:</span>
                  <span>{systemMetrics.disk_io.write_mb.toFixed(0)} MB</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Disk Usage Section */}
      {diskUsage && diskUsage.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-slate-100">Disk Usage</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {diskUsage.map((disk) => (
              <div key={disk.mount_point} className="bg-slate-900 rounded-lg p-4 border border-slate-800">
                <div className="text-sm text-slate-400 mb-2">{disk.mount_point}</div>
                <div className="text-2xl font-bold text-slate-100 mb-1">{disk.percent.toFixed(1)}%</div>
                <div className="text-xs text-slate-400 mb-2">
                  {disk.used_gb.toFixed(1)} / {disk.total_gb.toFixed(1)} GB
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2">
                  <div
                    className="bg-indigo-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(disk.percent, 100)}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      </div>
    </div>
  );
}
