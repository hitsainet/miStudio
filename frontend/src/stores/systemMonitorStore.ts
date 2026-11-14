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
import { devtools, persist } from 'zustand/middleware';
import type {
  GPUMetrics,
  GPUInfo,
  GPUProcess,
  SystemMetrics,
  DiskUsage,
  NetworkRates,
  DiskRates,
  AllMonitoringDataResponse,
  GPUListResponse,
  MonitoringGpuCard,
} from '../types/system';
import { getAllMonitoringData, getGPUList, getAllGPUMetrics } from '../api/system';

interface SystemMonitorState {
  // State
  gpuAvailable: boolean;
  gpuList: GPUListResponse | null;
  gpuMetrics: GPUMetrics | null;
  gpuInfo: GPUInfo | null;
  gpuProcesses: GPUProcess[];
  // Per-GPU metrics for comparison view
  allGpuMetrics: Record<number, GPUMetrics>;
  allGpuInfo: Record<number, GPUInfo>;
  systemMetrics: SystemMetrics | null;
  diskUsage: DiskUsage[];
  networkRates: NetworkRates | null;
  diskRates: DiskRates | null;
  /**
   * The card the single-card view shows, or null before any card is known.
   *
   * NOT DEFAULTED TO 0. Index 0 is whichever card sits first on the PCI bus;
   * adding a card on 2026-09-13 made that the 12 GB 3080 Ti instead of the
   * 3090, so a default of 0 silently changed which card the page described.
   */
  selectedGPU: number | null;
  updateInterval: number;
  viewMode: 'single' | 'compare';
  loading: boolean;
  error: string | null;
  errorType: 'connection' | 'gpu' | 'api' | 'general' | null;
  isPolling: boolean;
  isConnected: boolean;
  isWebSocketConnected: boolean; // NEW: WebSocket connection state
  consecutiveErrors: number;
  lastSuccessfulFetch: number | null;

  // Actions
  fetchGPUList: () => Promise<void>;
  fetchAllMetrics: () => Promise<void>;
  fetchAllGpuMetrics: () => Promise<void>;
  setSelectedGPU: (gpuId: number) => void;
  setUpdateInterval: (interval: number) => void;
  setViewMode: (mode: 'single' | 'compare') => void;
  startPolling: (interval?: number) => void;
  stopPolling: () => void;
  setError: (error: string | null, errorType?: 'connection' | 'gpu' | 'api' | 'general') => void;
  clearError: () => void;
  retryConnection: () => Promise<void>;
  // NEW: WebSocket update methods
  setIsWebSocketConnected: (connected: boolean) => void;
  setGPUMetrics: (gpuId: number, metrics: any) => void;
  updateSystemMetrics: (metrics: Partial<SystemMetrics>) => void;
}

/**
 * Every card in a `/system/all` response, whichever shape it came back in.
 *
 * Without a `gpu_id` the server lists every card under `gpu.cards`; with one it
 * returns that single card flat. Both are normalised to a list here so the rest
 * of the store has one shape to read.
 */
export function cardsOf(data: AllMonitoringDataResponse): MonitoringGpuCard[] {
  const gpu = data.gpu;
  if (!gpu) return [];
  if ('cards' in gpu) return gpu.cards;
  return [
    {
      gpu_id: gpu.selected_gpu_id,
      metrics: gpu.metrics,
      info: gpu.info,
      processes: gpu.processes,
    },
  ];
}

/**
 * The card the single-card view should show: the chosen one if it is still
 * listed, otherwise the first card listed. Never an id the server did not list.
 */
export function pickDisplayedCard(
  cards: MonitoringGpuCard[],
  selected: number | null,
): MonitoringGpuCard | null {
  if (cards.length === 0) return null;
  if (selected !== null) {
    const chosen = cards.find((card) => card.gpu_id === selected);
    if (chosen) return chosen;
  }
  return cards[0];
}

// Polling interval ID (stored outside of Zustand state)
let pollingIntervalId: number | null = null;

export const useSystemMonitorStore = create<SystemMonitorState>()(
  persist(
    devtools(
      (set, get) => ({
        // Initial state
        gpuAvailable: false,
        gpuList: null,
        gpuMetrics: null,
        gpuInfo: null,
        gpuProcesses: [],
        allGpuMetrics: {},
        allGpuInfo: {},
        systemMetrics: null,
        diskUsage: [],
        networkRates: null,
        diskRates: null,
        selectedGPU: null,
        updateInterval: 1000,
        viewMode: 'single',
        loading: false,
        error: null,
        errorType: null,
        isPolling: false,
        isConnected: true,
        isWebSocketConnected: false, // NEW: Initially false, set by WebSocket hook
        consecutiveErrors: 0,
        lastSuccessfulFetch: null,

        // Fetch GPU list
        fetchGPUList: async () => {
          try {
            const data = await getGPUList();
            set({ gpuList: data });
            // Also populate allGpuInfo from the list
            const allGpuInfo: Record<number, GPUInfo> = {};
            data.gpus.forEach((gpu: any) => {
              allGpuInfo[gpu.gpu_id] = gpu;
            });
            set({ allGpuInfo });
          } catch (error) {
            console.error('Failed to fetch GPU list:', error);
          }
        },

        // Fetch metrics for all GPUs (for comparison view)
        fetchAllGpuMetrics: async () => {
          try {
            const data = await getAllGPUMetrics();
            const allGpuMetrics: Record<number, GPUMetrics> = {};
            data.gpus.forEach((gpu: any) => {
              allGpuMetrics[gpu.gpu_id] = gpu;
            });
            set({ allGpuMetrics, lastSuccessfulFetch: Date.now() });
          } catch (error) {
            console.error('Failed to fetch all GPU metrics:', error);
          }
        },

      // Fetch all monitoring data in a single API call
      fetchAllMetrics: async () => {
        const { consecutiveErrors } = get();
        set({ loading: true, error: null, errorType: null });

        try {
          // NO CARD ID. `/system/all` without one returns every card, so the
          // store never names a card it has not seen listed — a persisted or
          // defaulted id that no longer exists would otherwise be a 400.
          const data: AllMonitoringDataResponse = await getAllMonitoringData();
          const cards = cardsOf(data);
          const shown = pickDisplayedCard(cards, get().selectedGPU);

          const allGpuMetrics: Record<number, GPUMetrics> = { ...get().allGpuMetrics };
          const allGpuInfo: Record<number, GPUInfo> = { ...get().allGpuInfo };
          cards.forEach((card) => {
            allGpuMetrics[card.gpu_id] = card.metrics;
            allGpuInfo[card.gpu_id] = card.info;
          });

          set({
            gpuAvailable: data.gpu_available,
            systemMetrics: data.system,
            diskUsage: data.disk_usage,
            networkRates: data.network_rates,
            diskRates: data.disk_rates,
            selectedGPU: shown ? shown.gpu_id : null,
            gpuMetrics: shown?.metrics ?? null,
            gpuInfo: shown?.info ?? null,
            gpuProcesses: shown?.processes ?? [],
            allGpuMetrics,
            allGpuInfo,
            loading: false,
            isConnected: true,
            consecutiveErrors: 0,
            lastSuccessfulFetch: Date.now(),
          });
        } catch (error) {
          const newConsecutiveErrors = consecutiveErrors + 1;
          const errorMessage =
            error instanceof Error ? error.message : 'Failed to fetch monitoring data';

          // Determine error type
          let errorType: 'connection' | 'gpu' | 'api' | 'general' = 'general';
          if (errorMessage.includes('Network') || errorMessage.includes('fetch')) {
            errorType = 'connection';
          } else if (errorMessage.includes('GPU') || errorMessage.includes('CUDA')) {
            errorType = 'gpu';
          } else if (errorMessage.includes('API') || errorMessage.includes('status')) {
            errorType = 'api';
          }

          set({
            error: errorMessage,
            errorType,
            loading: false,
            isConnected: errorType !== 'connection',
            consecutiveErrors: newConsecutiveErrors,
          });

          // Stop polling if too many consecutive errors (>5)
          if (newConsecutiveErrors > 5 && get().isPolling) {
            console.error('Too many consecutive errors, stopping polling');
            get().stopPolling();
          }
        }
      },

      // Set selected GPU
      setSelectedGPU: (gpuId: number) => {
        set({ selectedGPU: gpuId });
        // Fetch metrics for newly selected GPU
        get().fetchAllMetrics();
      },

      // Set update interval
      setUpdateInterval: (interval: number) => {
        set({ updateInterval: interval });
        // Restart polling with new interval if currently polling
        const { isPolling, stopPolling, startPolling } = get();
        if (isPolling) {
          stopPolling();
          startPolling(interval);
        }
      },

      // Set view mode
      setViewMode: (mode: 'single' | 'compare') => {
        set({ viewMode: mode });
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
        // Only update state if polling was actually active
        if (get().isPolling) {
          set({ isPolling: false });
        }
      },

      // Set error message
      setError: (error: string | null, errorType: 'connection' | 'gpu' | 'api' | 'general' = 'general') => {
        set({ error, errorType });
      },

      // Clear error message
      clearError: () => {
        set({ error: null, errorType: null, consecutiveErrors: 0 });
      },

      // Retry connection
      retryConnection: async () => {
        const { isPolling, updateInterval, startPolling } = get();
        set({ error: null, errorType: null, consecutiveErrors: 0 });

        if (!isPolling) {
          // Restart polling
          startPolling(updateInterval);
        } else {
          // Just fetch once
          await get().fetchAllMetrics();
        }
      },

      // NEW: WebSocket update methods
      // Set WebSocket connection state
      setIsWebSocketConnected: (connected: boolean) => {
        set({ isWebSocketConnected: connected });

        // If WebSocket disconnected, start polling as fallback
        if (!connected && !get().isPolling) {
          console.log('[SystemMonitorStore] WebSocket disconnected, starting polling fallback');
          get().startPolling(get().updateInterval);
        }
        // If WebSocket connected, stop polling (WebSocket will provide updates)
        else if (connected && get().isPolling) {
          console.log('[SystemMonitorStore] WebSocket connected, stopping polling');
          get().stopPolling();
        }
      },

      // Set GPU metrics from WebSocket update
      setGPUMetrics: (gpuId: number, metrics: any) => {
        const state = get();
        // Always update allGpuMetrics for comparison view
        const newAllGpuMetrics = { ...state.allGpuMetrics, [gpuId]: metrics };

        // Update selected GPU metrics if this is the selected GPU
        if (gpuId === state.selectedGPU) {
          set({
            gpuMetrics: metrics,
            allGpuMetrics: newAllGpuMetrics,
            lastSuccessfulFetch: Date.now(),
            consecutiveErrors: 0,
          });
        } else {
          set({
            allGpuMetrics: newAllGpuMetrics,
            lastSuccessfulFetch: Date.now(),
          });
        }
      },

      // Update system metrics from WebSocket (partial update)
      updateSystemMetrics: (metrics: Partial<SystemMetrics>) => {
        set((state) => ({
          systemMetrics: {
            ...state.systemMetrics,
            ...metrics,
          } as SystemMetrics,
          lastSuccessfulFetch: Date.now(),
          consecutiveErrors: 0,
        }));
      },
      }),
      {
        name: 'SystemMonitorStore',
      }
    ),
    {
      name: 'system-monitor-settings',
      partialize: (state) => ({
        updateInterval: state.updateInterval,
        viewMode: state.viewMode,
        selectedGPU: state.selectedGPU,
      }),
    }
  )
);
