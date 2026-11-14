/**
 * Zustand store for Neuronpedia Export state management.
 */

import { create } from 'zustand';
import {
  NeuronpediaExportConfig,
  NeuronpediaExportJob,
  ExportProgressEvent,
  DEFAULT_EXPORT_CONFIG,
} from '../types/neuronpedia';
import * as neuronpediaApi from '../api/neuronpedia';

interface NeuronpediaExportState {
  // Current export configuration
  config: NeuronpediaExportConfig;

  // Active export job (if any)
  currentJob: NeuronpediaExportJob | null;

  // All export jobs for the current SAE
  exportJobs: NeuronpediaExportJob[];

  // UI state
  isExportDialogOpen: boolean;
  selectedSaeId: string | null;
  isExporting: boolean;
  isLoading: boolean;
  error: string | null;

  // Polling state
  pollingInterval: ReturnType<typeof setInterval> | null;

  // Actions
  setConfig: (config: Partial<NeuronpediaExportConfig>) => void;
  resetConfig: () => void;

  openExportDialog: (saeId: string) => void;
  closeExportDialog: () => void;

  startExport: () => Promise<void>;
  cancelExport: () => Promise<void>;
  deleteExport: (jobId: string) => Promise<void>;

  fetchExportJobs: (saeId?: string) => Promise<void>;
  fetchJobStatus: (jobId: string) => Promise<void>;

  downloadExport: (jobId: string, filename?: string) => Promise<void>;

  // WebSocket handling
  handleProgressEvent: (event: ExportProgressEvent) => void;

  // Polling
  startPolling: (jobId: string) => void;
  stopPolling: () => void;

  // Reset
  reset: () => void;
}

export const useNeuronpediaExportStore = create<NeuronpediaExportState>((set, get) => ({
  // Initial state
  config: { ...DEFAULT_EXPORT_CONFIG },
  currentJob: null,
  exportJobs: [],
  isExportDialogOpen: false,
  selectedSaeId: null,
  isExporting: false,
  isLoading: false,
  error: null,
  pollingInterval: null,

  // Config actions
  setConfig: (partialConfig) => {
    set((state) => ({
      config: { ...state.config, ...partialConfig },
    }));
  },

  resetConfig: () => {
    set({ config: { ...DEFAULT_EXPORT_CONFIG } });
  },

  // Dialog actions
  openExportDialog: (saeId) => {
    set({
      isExportDialogOpen: true,
      selectedSaeId: saeId,
      error: null,
      currentJob: null,
    });
    // Fetch existing export jobs for this SAE
    get().fetchExportJobs(saeId);
  },

  closeExportDialog: () => {
    get().stopPolling();
    set({
      isExportDialogOpen: false,
      selectedSaeId: null,
      currentJob: null,
      error: null,
    });
  },

  // Export actions
  startExport: async () => {
    const { selectedSaeId, config } = get();
    if (!selectedSaeId) {
      set({ error: 'No SAE selected for export' });
      return;
    }

    set({ isExporting: true, error: null });

    try {
      const response = await neuronpediaApi.startExport(selectedSaeId, config);

      // Fetch the full job status
      const job = await neuronpediaApi.getExportStatus(response.jobId);

      set({
        currentJob: job,
        isExporting: false,
      });

      // Start polling for updates
      get().startPolling(response.jobId);
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Failed to start export',
        isExporting: false,
      });
    }
  },

  cancelExport: async () => {
    const { currentJob } = get();
    if (!currentJob) return;

    try {
      await neuronpediaApi.cancelExport(currentJob.id);
      get().stopPolling();

      // Update job status locally
      set({
        currentJob: { ...currentJob, status: 'cancelled' },
      });

      // Refresh job list
      const { selectedSaeId } = get();
      if (selectedSaeId) {
        get().fetchExportJobs(selectedSaeId);
      }
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Failed to cancel export',
      });
    }
  },

  deleteExport: async (jobId) => {
    try {
      await neuronpediaApi.deleteExport(jobId);

      // Remove from local state
      set((state) => ({
        exportJobs: state.exportJobs.filter((j) => j.id !== jobId),
        currentJob: state.currentJob?.id === jobId ? null : state.currentJob,
      }));
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Failed to delete export',
      });
    }
  },

  // Fetch actions
  fetchExportJobs: async (saeId) => {
    const targetSaeId = saeId || get().selectedSaeId;
    if (!targetSaeId) return;

    set({ isLoading: true });

    try {
      const response = await neuronpediaApi.listExports({ saeId: targetSaeId });
      set({
        exportJobs: response.jobs,
        isLoading: false,
      });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Failed to fetch export jobs',
        isLoading: false,
      });
    }
  },

  fetchJobStatus: async (jobId) => {
    try {
      const job = await neuronpediaApi.getExportStatus(jobId);
      set({ currentJob: job });

      // Update in jobs list too
      set((state) => ({
        exportJobs: state.exportJobs.map((j) => (j.id === jobId ? job : j)),
      }));

      // Stop polling if job is complete
      if (['completed', 'failed', 'cancelled'].includes(job.status)) {
        get().stopPolling();
      }
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Failed to fetch job status',
      });
      get().stopPolling();
    }
  },

  // Download action
  downloadExport: async (jobId, filename) => {
    try {
      const blob = await neuronpediaApi.downloadExport(jobId);
      const downloadFilename = filename || `neuronpedia-export-${jobId}.zip`;
      neuronpediaApi.triggerDownload(blob, downloadFilename);
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Failed to download export',
      });
    }
  },

  // WebSocket handling
  handleProgressEvent: (event) => {
    const { currentJob } = get();
    if (!currentJob || currentJob.id !== event.jobId) return;

    set({
      currentJob: {
        ...currentJob,
        progress: event.progress,
        currentStage: event.stage,
        status: event.status,
        featureCount: event.featureCount ?? currentJob.featureCount,
        outputPath: event.outputPath ?? currentJob.outputPath,
      },
    });

    // Stop polling if complete
    if (['completed', 'failed', 'cancelled'].includes(event.status)) {
      get().stopPolling();
    }
  },

  // Polling
  startPolling: (jobId) => {
    // Stop any existing polling
    get().stopPolling();

    // Poll every 2 seconds
    const interval = setInterval(() => {
      get().fetchJobStatus(jobId);
    }, 2000);

    set({ pollingInterval: interval });
  },

  stopPolling: () => {
    const { pollingInterval } = get();
    if (pollingInterval) {
      clearInterval(pollingInterval);
      set({ pollingInterval: null });
    }
  },

  // Reset
  reset: () => {
    get().stopPolling();
    set({
      config: { ...DEFAULT_EXPORT_CONFIG },
      currentJob: null,
      exportJobs: [],
      isExportDialogOpen: false,
      selectedSaeId: null,
      isExporting: false,
      isLoading: false,
      error: null,
      pollingInterval: null,
    });
  },
}));

// Selector helpers
export const selectIsExportInProgress = (state: NeuronpediaExportState) =>
  state.currentJob?.status === 'computing' || state.currentJob?.status === 'packaging';

export const selectIsExportComplete = (state: NeuronpediaExportState) =>
  state.currentJob?.status === 'completed';

export const selectCanDownload = (state: NeuronpediaExportState) =>
  state.currentJob?.status === 'completed' && state.currentJob?.downloadUrl;

export const selectProgress = (state: NeuronpediaExportState) =>
  state.currentJob?.progress ?? 0;

export const selectCurrentStage = (state: NeuronpediaExportState) =>
  state.currentJob?.currentStage ?? '';
