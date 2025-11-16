/**
 * System Monitor Page
 *
 * Main page component for system and GPU monitoring dashboard.
 * Displays real-time metrics with auto-refresh.
 */

import { useEffect, useState, useMemo, useRef } from 'react';
import { Activity, Settings } from 'lucide-react';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';
import { useHistoricalData } from '../../hooks/useHistoricalData';
import { useSystemMonitorWebSocket } from '../../hooks/useSystemMonitorWebSocket';
import { UtilizationChart } from './UtilizationChart';
import { MemoryUsageChart } from './MemoryUsageChart';
import { GPUSelector } from './GPUSelector';
import { ViewModeToggle } from './ViewModeToggle';
import { GPUCard } from './GPUCard';
import { SettingsModal } from './SettingsModal';
import { LoadingSkeleton } from './LoadingSkeleton';
import { ErrorBanner } from './ErrorBanner';
import { MetricWarning } from './MetricWarning';
import { ActiveOperationsSection } from './ActiveOperationsSection';
import { FailedOperationsSection } from './FailedOperationsSection';

export function SystemMonitor() {
  const {
    gpuAvailable,
    gpuList,
    gpuMetrics,
    gpuInfo,
    gpuProcesses,
    systemMetrics,
    diskUsage,
    selectedGPU,
    updateInterval,
    viewMode,
    loading,
    error,
    errorType,
    isPolling,
    isConnected,
    isWebSocketConnected,
    fetchGPUList,
    setSelectedGPU,
    setUpdateInterval,
    setViewMode,
    startPolling,
    stopPolling,
    clearError,
    retryConnection,
  } = useSystemMonitorStore();

  // Settings modal state
  const [showSettings, setShowSettings] = useState(false);

  // Initialize historical data hook - fixed to 1 hour
  const {
    data: historicalData,
    addDataPoint,
  } = useHistoricalData({ maxDataPoints: 3600 }); // 1h at 1s intervals

  // Get GPU IDs for WebSocket subscription
  const gpuIds = useMemo(() => {
    if (!gpuList) return [];
    return gpuList.gpus.map(gpu => gpu.gpu_id);
  }, [gpuList]);

  // Subscribe to system monitoring WebSocket channels
  useSystemMonitorWebSocket(gpuIds);

  // Fetch GPU list on mount
  useEffect(() => {
    fetchGPUList();
  }, [fetchGPUList]);

  // Fallback polling: Only start if WebSocket is not connected
  // WebSocket connection state managed by useSystemMonitorWebSocket hook
  const hasStartedPolling = useRef(false);

  useEffect(() => {
    // Start polling if WebSocket is not connected and we haven't started polling yet
    if (!isWebSocketConnected && !hasStartedPolling.current) {
      console.log('[SystemMonitor] Starting polling fallback (WebSocket not connected)');
      startPolling(updateInterval);
      hasStartedPolling.current = true;
    }
    // Stop polling if WebSocket connects and we had started polling
    else if (isWebSocketConnected && hasStartedPolling.current) {
      console.log('[SystemMonitor] Stopping polling (WebSocket connected)');
      stopPolling();
      hasStartedPolling.current = false;
    }

    return () => {
      // Clean up polling only on unmount, and only if we started it
      if (hasStartedPolling.current) {
        console.log('[SystemMonitor] Component unmounting, stopping polling');
        stopPolling();
        hasStartedPolling.current = false;
      }
    };
  }, [isWebSocketConnected, updateInterval, startPolling, stopPolling]);

  // Update historical data when new metrics arrive
  useEffect(() => {
    if (gpuMetrics && systemMetrics) {
      addDataPoint({
        gpu_utilization: gpuMetrics.utilization.gpu,
        cpu_utilization: systemMetrics.cpu.percent,
        gpu_memory_used_gb: gpuMetrics.memory.used_gb,
        ram_used_gb: systemMetrics.ram.used_gb,
        gpu_temperature: gpuMetrics.temperature,
      });
    }
  }, [gpuMetrics, systemMetrics, addDataPoint]);

  if (loading && !systemMetrics) {
    return <LoadingSkeleton />;
  }

  return (
    <div className="min-h-screen bg-slate-950 animate-in fade-in duration-500">
      <div className="max-w-[80%] mx-auto px-6 py-8 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-8 h-8 text-emerald-400" />
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">System Monitor</h1>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                Real-time GPU and system resource monitoring
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {isPolling && (
              <div className="flex items-center gap-2 text-sm text-emerald-400">
                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                <span>Live</span>
              </div>
            )}
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 rounded-lg border border-slate-700 bg-slate-900 hover:bg-slate-800 transition-colors"
              aria-label="Settings"
              title="Configure system monitor settings"
            >
              <Settings className="w-5 h-5 text-slate-400" />
            </button>
          </div>
        </div>

        {/* Error Banner */}
        {error && errorType && (
          <ErrorBanner
            type={errorType}
            message={error}
            isRetrying={loading}
            onRetry={retryConnection}
            onDismiss={!isConnected ? undefined : clearError}
          />
        )}

      {/* Main Layout: System Resources (Left) + GPU Information (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* LEFT COLUMN: System Resources */}
        {systemMetrics && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-slate-100">System Resources</h2>

            {/* CPU */}
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
              <div className="text-sm text-slate-400 mb-2">CPU Utilization</div>
              <div className="text-3xl font-bold text-slate-100 mb-1">
                {systemMetrics.cpu.percent.toFixed(1)}%
              </div>
              <div className="text-xs text-slate-400 mb-2">{systemMetrics.cpu.count} cores (max {systemMetrics.cpu.count * 100}%)</div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div
                  className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min((systemMetrics.cpu.percent / (systemMetrics.cpu.count * 100)) * 100, 100)}%` }}
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

            {/* Disk Usage */}
            {diskUsage && diskUsage.length > 0 && diskUsage.map((disk) => (
              <div key={disk.mount_point} className="bg-slate-900 rounded-lg p-4 border border-slate-800">
                <div className="text-sm text-slate-400 mb-2">{disk.mount_point}</div>
                <div className="text-3xl font-bold text-slate-100 mb-1">{disk.percent.toFixed(1)}%</div>
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

            {/* Disk I/O */}
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
              <div className="text-sm text-slate-400 mb-2">Disk I/O</div>
              <div className="text-sm text-slate-100">
                <div className="flex justify-between mb-1">
                  <span className="text-slate-600 dark:text-slate-400">Read:</span>
                  <span>{systemMetrics.disk_io.read_mb.toFixed(0)} MB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600 dark:text-slate-400">Write:</span>
                  <span>{systemMetrics.disk_io.write_mb.toFixed(0)} MB</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* RIGHT COLUMN: GPU Information */}
        {gpuAvailable && gpuMetrics && gpuInfo && viewMode === 'single' ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-slate-100">GPU Information</h2>
              <div className="flex items-center gap-2">
                {gpuList && viewMode === 'single' && gpuList.gpus.length > 0 && (
                  <GPUSelector
                    gpus={gpuList.gpus}
                    selected={selectedGPU}
                    onChange={setSelectedGPU}
                  />
                )}
                {gpuList && gpuList.gpus.length > 1 && (
                  <ViewModeToggle mode={viewMode} onChange={setViewMode} />
                )}
              </div>
            </div>

            {/* Critical Warnings */}
            {(gpuMetrics.temperature > 85 || gpuMetrics.memory.used_percent > 95 || gpuMetrics.utilization.gpu > 95) && (
              <div className="flex flex-wrap gap-3">
                <MetricWarning type="temperature" value={gpuMetrics.temperature} />
                <MetricWarning type="memory" value={gpuMetrics.memory.used_percent} />
                <MetricWarning type="utilization" value={gpuMetrics.utilization.gpu} />
              </div>
            )}

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

            {/* GPU Device Info */}
            <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-slate-600 dark:text-slate-400">Device</div>
                  <div className="text-slate-100 font-medium">{gpuInfo.name}</div>
                </div>
                <div>
                  <div className="text-slate-600 dark:text-slate-400">Driver</div>
                  <div className="text-slate-100 font-medium">{gpuInfo.driver_version}</div>
                </div>
                <div>
                  <div className="text-slate-600 dark:text-slate-400">CUDA</div>
                  <div className="text-slate-100 font-medium">{gpuInfo.cuda_version}</div>
                </div>
                <div>
                  <div className="text-slate-600 dark:text-slate-400">Memory</div>
                  <div className="text-slate-100 font-medium">{gpuInfo.total_memory_gb} GB</div>
                </div>
              </div>
            </div>
          </div>
        ) : gpuAvailable && !gpuMetrics ? (
          <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
            <div className="text-slate-400 text-center">
              GPU monitoring not available. Ensure NVIDIA GPU and drivers are installed.
            </div>
          </div>
        ) : null}
      </div>

      {/* GPU Processes - Full Width */}
      {gpuAvailable && gpuMetrics && gpuInfo && viewMode === 'single' && gpuProcesses && gpuProcesses.length > 0 && (
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

      {/* GPU Comparison View (when multiple GPUs) */}
      {gpuAvailable && viewMode === 'compare' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h2 className="text-lg font-semibold text-slate-100">GPU Comparison</h2>
              {gpuList && gpuList.gpus.length > 4 && (
                <span className="text-sm text-slate-600 dark:text-slate-400">
                  {gpuList.gpus.length} GPUs detected - Scroll to view all
                </span>
              )}
            </div>
            {gpuList && gpuList.gpus.length > 1 && (
              <ViewModeToggle mode={viewMode} onChange={setViewMode} />
            )}
          </div>
          <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 ${
            gpuList && gpuList.gpus.length > 6 ? 'max-h-[800px] overflow-y-auto pr-2' : ''
          }`}>
            {gpuList?.gpus.map((gpu) => (
              <GPUCard
                key={gpu.gpu_id}
                gpuId={gpu.gpu_id}
                metrics={gpuMetrics || null}
                info={gpuInfo || null}
              />
            ))}
          </div>
        </div>
      )}

      {/* Historical Trends - Full Width */}
      {gpuAvailable && gpuMetrics && gpuInfo && viewMode === 'single' && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-slate-100">Historical Trends (Last Hour)</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <UtilizationChart data={historicalData} />
            <MemoryUsageChart data={historicalData} />
          </div>
        </div>
      )}

      {/* Task Queue Operations */}
      <div className="space-y-6">
        <ActiveOperationsSection />
        <FailedOperationsSection />
      </div>

      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        updateInterval={updateInterval}
        onUpdateIntervalChange={setUpdateInterval}
      />
    </div>
  );
}
