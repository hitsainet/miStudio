/**
 * Zustand store for System Monitor.
 *
 * This store manages the global state for system and GPU monitoring, including:
 * - GPU metrics (utilization, memory, temperature, power, etc.)
 * - System metrics (CPU, RAM, swap, disk I/O, network I/O)
 * - GPU information and processes
 * - Auto-refresh with configurable interval
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type {
  GPUMetrics,
  GPUInfo,
  GPUProcess,
  SystemMetrics,
  DiskUsage,
  NetworkRates,
  DiskRates,
  AllMonitoringDataResponse,
} from '../types/system';
import { getAllMonitoringData } from '../api/system';

interface SystemMonitorState {
  // State
  gpuAvailable: boolean;
  gpuMetrics: GPUMetrics | null;
  gpuInfo: GPUInfo | null;
  gpuProcesses: GPUProcess[];
  systemMetrics: SystemMetrics | null;
  diskUsage: DiskUsage[];
  networkRates: NetworkRates | null;
  diskRates: DiskRates | null;
  selectedGPU: number;
  loading: boolean;
  error: string | null;
  isPolling: boolean;

  // Actions
  fetchAllMetrics: () => Promise<void>;
  setSelectedGPU: (gpuId: number) => void;
  startPolling: (interval?: number) => void;
  stopPolling: () => void;
  setError: (error: string | null) => void;
  clearError: () => void;
}

// Polling interval ID (stored outside of Zustand state)
let pollingIntervalId: NodeJS.Timeout | null = null;

export const useSystemMonitorStore = create<SystemMonitorState>()(
  devtools(
    (set, get) => ({
      // Initial state
      gpuAvailable: false,
      gpuMetrics: null,
      gpuInfo: null,
      gpuProcesses: [],
      systemMetrics: null,
      diskUsage: [],
      networkRates: null,
      diskRates: null,
      selectedGPU: 0,
      loading: false,
      error: null,
      isPolling: false,

      // Fetch all monitoring data in a single API call
      fetchAllMetrics: async () => {
        const { selectedGPU } = get();
        set({ loading: true, error: null });

        try {
          const data: AllMonitoringDataResponse = await getAllMonitoringData(selectedGPU);

          set({
            gpuAvailable: data.gpu_available,
            systemMetrics: data.system,
            diskUsage: data.disk_usage,
            networkRates: data.network_rates,
            diskRates: data.disk_rates,
            gpuMetrics: data.gpu?.metrics || null,
            gpuInfo: data.gpu?.info || null,
            gpuProcesses: data.gpu?.processes || [],
            loading: false,
          });
        } catch (error) {
          const errorMessage =
            error instanceof Error ? error.message : 'Failed to fetch monitoring data';
          set({ error: errorMessage, loading: false });
        }
      },

      // Set selected GPU
      setSelectedGPU: (gpuId: number) => {
        set({ selectedGPU: gpuId });
        // Fetch metrics for newly selected GPU
        get().fetchAllMetrics();
      },

      // Start auto-refresh polling
      startPolling: (interval: number = 1000) => {
        // Stop existing polling if any
        get().stopPolling();

        // Set polling state
        set({ isPolling: true });

        // Initial fetch
        get().fetchAllMetrics();

        // Start interval
        pollingIntervalId = setInterval(() => {
          get().fetchAllMetrics();
        }, interval);
      },

      // Stop auto-refresh polling
      stopPolling: () => {
        if (pollingIntervalId) {
          clearInterval(pollingIntervalId);
          pollingIntervalId = null;
        }
        set({ isPolling: false });
      },

      // Set error message
      setError: (error: string | null) => {
        set({ error });
      },

      // Clear error message
      clearError: () => {
        set({ error: null });
      },
    }),
    {
      name: 'SystemMonitorStore',
    }
  )
);
