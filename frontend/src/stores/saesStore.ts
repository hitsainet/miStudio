/**
 * Zustand store for SAE management.
 *
 * This store manages the global state for SAEs (Sparse Autoencoders), including:
 * - List of SAEs (downloaded from HuggingFace, imported from training, local files)
 * - Loading states
 * - Error handling
 * - CRUD operations (download, import, delete)
 * - HuggingFace repository preview
 * - Real-time download progress updates
 *
 * Connects to REAL backend API at /api/v1/saes
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import {
  SAE,
  SAESource,
  SAEStatus,
  HFRepoPreviewRequest,
  HFRepoPreviewResponse,
  SAEDownloadRequest,
  SAEUploadRequest,
  SAEUploadResponse,
  SAEImportFromTrainingRequest,
  SAEImportFromFileRequest,
  SAEFeatureBrowserResponse,
  SAEDeleteRequest,
} from '../types/sae';
import * as saesApi from '../api/saes';

// Callback for subscribing to SAE download progress (set by WebSocket context)
let subscribeToSAECallback: ((saeId: string) => void) | null = null;

export function setSAESubscriptionCallback(callback: (saeId: string) => void) {
  subscribeToSAECallback = callback;
}

interface SAEsState {
  // State
  saes: SAE[];
  loading: boolean;
  error: string | null;

  // HuggingFace preview state
  hfPreview: HFRepoPreviewResponse | null;
  hfPreviewLoading: boolean;
  hfPreviewError: string | null;

  // Feature browser state
  featureBrowser: {
    saeId: string | null;
    data: SAEFeatureBrowserResponse | null;
    loading: boolean;
    error: string | null;
  };

  // Filters
  filters: {
    search: string;
    source: SAESource | null;
    status: SAEStatus | null;
    modelName: string | null;
    sortBy: string;
    order: 'asc' | 'desc';
  };

  // Pagination
  pagination: {
    skip: number;
    limit: number;
    total: number;
    hasMore: boolean;
  };

  // Actions - Core CRUD
  fetchSAEs: () => Promise<void>;
  getSAE: (id: string) => Promise<SAE>;
  deleteSAE: (id: string, deleteFiles?: boolean) => Promise<void>;
  deleteSAEsBatch: (ids: string[], deleteFiles?: boolean) => Promise<void>;

  // Actions - HuggingFace
  previewHFRepository: (request: HFRepoPreviewRequest) => Promise<HFRepoPreviewResponse>;
  clearHFPreview: () => void;
  downloadSAE: (request: SAEDownloadRequest) => Promise<SAE>;
  uploadSAE: (request: SAEUploadRequest) => Promise<SAEUploadResponse>;

  // Actions - Import
  importFromTraining: (request: SAEImportFromTrainingRequest) => Promise<SAE>;
  importFromFile: (request: SAEImportFromFileRequest) => Promise<SAE>;

  // Actions - Feature Browser
  browseFeatures: (saeId: string, params?: { skip?: number; limit?: number; search?: string }) => Promise<void>;
  clearFeatureBrowser: () => void;

  // Actions - Progress Updates (WebSocket)
  updateDownloadProgress: (id: string, progress: number, status?: SAEStatus, message?: string) => void;
  updateSAEStatus: (id: string, status: SAEStatus, errorMessage?: string) => void;

  // Actions - Filters & Pagination
  setFilters: (filters: Partial<SAEsState['filters']>) => void;
  setPage: (skip: number) => void;

  // Actions - Error Handling
  setError: (error: string | null) => void;
  clearError: () => void;
}

export const useSAEsStore = create<SAEsState>()(
  devtools(
    (set, get) => ({
      // Initial state
      saes: [],
      loading: false,
      error: null,

      hfPreview: null,
      hfPreviewLoading: false,
      hfPreviewError: null,

      featureBrowser: {
        saeId: null,
        data: null,
        loading: false,
        error: null,
      },

      filters: {
        search: '',
        source: null,
        status: null,
        modelName: null,
        sortBy: 'created_at',
        order: 'desc',
      },

      pagination: {
        skip: 0,
        limit: 20,
        total: 0,
        hasMore: false,
      },

      // Fetch all SAEs with current filters
      fetchSAEs: async () => {
        set({ loading: true, error: null });
        try {
          const { filters, pagination } = get();
          const params: Parameters<typeof saesApi.getSAEs>[0] = {
            skip: pagination.skip,
            limit: pagination.limit,
            sort_by: filters.sortBy,
            order: filters.order,
          };

          if (filters.search) params.search = filters.search;
          if (filters.source) params.source = filters.source;
          if (filters.status) params.status = filters.status;
          if (filters.modelName) params.model_name = filters.modelName;

          const response = await saesApi.getSAEs(params);

          set({
            saes: response.data,
            pagination: {
              skip: response.pagination.skip,
              limit: response.pagination.limit,
              total: response.pagination.total,
              hasMore: response.pagination.has_more,
            },
            loading: false,
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch SAEs';
          set({ error: errorMessage, loading: false });
        }
      },

      // Get a single SAE by ID
      getSAE: async (id: string) => {
        try {
          return await saesApi.getSAE(id);
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch SAE';
          set({ error: errorMessage });
          throw error;
        }
      },

      // Delete a single SAE
      deleteSAE: async (id: string, deleteFiles = true) => {
        set({ loading: true, error: null });
        try {
          await saesApi.deleteSAE(id, deleteFiles);
          set((state) => ({
            saes: state.saes.filter((s) => s.id !== id),
            pagination: {
              ...state.pagination,
              total: state.pagination.total - 1,
            },
            loading: false,
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to delete SAE';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Delete multiple SAEs
      deleteSAEsBatch: async (ids: string[], deleteFiles = true) => {
        set({ loading: true, error: null });
        try {
          const request: SAEDeleteRequest = { ids };
          await saesApi.deleteSAEsBatch(request, deleteFiles);
          set((state) => ({
            saes: state.saes.filter((s) => !ids.includes(s.id)),
            pagination: {
              ...state.pagination,
              total: state.pagination.total - ids.length,
            },
            loading: false,
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to delete SAEs';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Preview a HuggingFace repository
      previewHFRepository: async (request: HFRepoPreviewRequest) => {
        set({ hfPreviewLoading: true, hfPreviewError: null });
        try {
          const response = await saesApi.previewHFRepository(request);
          set({ hfPreview: response, hfPreviewLoading: false });
          return response;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to preview repository';
          set({ hfPreviewError: errorMessage, hfPreviewLoading: false });
          throw error;
        }
      },

      // Clear HuggingFace preview
      clearHFPreview: () => {
        set({ hfPreview: null, hfPreviewError: null });
      },

      // Download an SAE from HuggingFace
      downloadSAE: async (request: SAEDownloadRequest) => {
        set({ loading: true, error: null });
        try {
          const newSAE = await saesApi.downloadSAE(request);

          // Subscribe to download progress updates
          if (subscribeToSAECallback && newSAE.id) {
            console.log('[SAEsStore] Subscribing to SAE download progress:', newSAE.id);
            subscribeToSAECallback(newSAE.id);
          }

          // Add SAE to store
          set((state) => ({
            saes: [newSAE, ...state.saes],
            pagination: {
              ...state.pagination,
              total: state.pagination.total + 1,
            },
            loading: false,
          }));

          return newSAE;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to download SAE';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Upload an SAE to HuggingFace
      uploadSAE: async (request: SAEUploadRequest) => {
        set({ loading: true, error: null });
        try {
          const response = await saesApi.uploadSAE(request);
          set({ loading: false });
          return response;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to upload SAE';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Import an SAE from a completed training job
      importFromTraining: async (request: SAEImportFromTrainingRequest) => {
        set({ loading: true, error: null });
        try {
          const newSAE = await saesApi.importSAEFromTraining(request);

          // Add SAE to store
          set((state) => ({
            saes: [newSAE, ...state.saes],
            pagination: {
              ...state.pagination,
              total: state.pagination.total + 1,
            },
            loading: false,
          }));

          return newSAE;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to import SAE from training';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Import an SAE from a local file
      importFromFile: async (request: SAEImportFromFileRequest) => {
        set({ loading: true, error: null });
        try {
          const newSAE = await saesApi.importSAEFromFile(request);

          // Add SAE to store
          set((state) => ({
            saes: [newSAE, ...state.saes],
            pagination: {
              ...state.pagination,
              total: state.pagination.total + 1,
            },
            loading: false,
          }));

          return newSAE;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to import SAE from file';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Browse features in an SAE
      browseFeatures: async (saeId: string, params?: { skip?: number; limit?: number; search?: string }) => {
        set((state) => ({
          featureBrowser: {
            ...state.featureBrowser,
            saeId,
            loading: true,
            error: null,
          },
        }));
        try {
          const response = await saesApi.browseSAEFeatures(saeId, params);
          set((state) => ({
            featureBrowser: {
              ...state.featureBrowser,
              data: response,
              loading: false,
            },
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to browse features';
          set((state) => ({
            featureBrowser: {
              ...state.featureBrowser,
              error: errorMessage,
              loading: false,
            },
          }));
        }
      },

      // Clear feature browser state
      clearFeatureBrowser: () => {
        set({
          featureBrowser: {
            saeId: null,
            data: null,
            loading: false,
            error: null,
          },
        });
      },

      // Update download progress (called by WebSocket)
      updateDownloadProgress: (id: string, progress: number, status?: SAEStatus, message?: string) => {
        set((state) => ({
          saes: state.saes.map((sae) =>
            sae.id === id
              ? {
                  ...sae,
                  progress,
                  status: status || sae.status,
                  error_message: message || sae.error_message,
                }
              : sae
          ),
        }));
      },

      // Update SAE status (called by WebSocket)
      updateSAEStatus: (id: string, status: SAEStatus, errorMessage?: string) => {
        set((state) => ({
          saes: state.saes.map((sae) =>
            sae.id === id
              ? {
                  ...sae,
                  status,
                  error_message: errorMessage ?? null,
                  progress: status === SAEStatus.READY ? 100 : sae.progress,
                }
              : sae
          ),
        }));
      },

      // Set filters
      setFilters: (filters: Partial<SAEsState['filters']>) => {
        set((state) => ({
          filters: { ...state.filters, ...filters },
          pagination: { ...state.pagination, skip: 0 }, // Reset to first page
        }));
      },

      // Set page
      setPage: (skip: number) => {
        set((state) => ({
          pagination: { ...state.pagination, skip },
        }));
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
      name: 'SAEsStore',
    }
  )
);

// Selector for ready SAEs (for steering)
export const selectReadySAEs = (state: SAEsState) =>
  state.saes.filter((sae) => sae.status === SAEStatus.READY);

// Selector for SAEs by model
export const selectSAEsByModel = (modelName: string) => (state: SAEsState) =>
  state.saes.filter((sae) => sae.model_name === modelName && sae.status === SAEStatus.READY);
