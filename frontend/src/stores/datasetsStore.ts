/**
 * Zustand store for dataset management.
 *
 * This store manages the global state for datasets, including:
 * - List of datasets
 * - Loading states
 * - Error handling
 * - CRUD operations
 * - Real-time progress updates via WebSocket
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { Dataset, DatasetStatus, DatasetTokenization, DatasetTokenizationListResponse } from '../types/dataset';
import { API_BASE_URL } from '../config/api';
import { cancelDatasetDownload, getDataset } from '../api/datasets';
import { startPolling } from '../utils/polling';

// Callback for subscribing to dataset progress (set by WebSocket context)
let subscribeToDatasetCallback: ((datasetId: string) => void) | null = null;

export function setDatasetSubscriptionCallback(callback: (datasetId: string) => void) {
  subscribeToDatasetCallback = callback;
}

interface DatasetsState {
  // State
  datasets: Dataset[];
  tokenizations: Record<string, DatasetTokenization[]>; // keyed by dataset_id
  loading: boolean;
  error: string | null;

  // Actions
  fetchDatasets: () => Promise<void>;
  downloadDataset: (repoId: string, accessToken?: string, split?: string, config?: string) => Promise<void>;
  deleteDataset: (id: string) => Promise<void>;
  cancelDownload: (id: string) => Promise<void>;
  updateDatasetProgress: (id: string, progress: number) => void;
  updateDatasetStatus: (id: string, status: DatasetStatus, errorMessage?: string) => void;
  setError: (error: string | null) => void;
  clearError: () => void;

  // Tokenization actions
  fetchTokenizations: (datasetId: string) => Promise<void>;
  createTokenization: (datasetId: string, modelId: string, params: any) => Promise<void>;
  deleteTokenization: (datasetId: string, modelId: string) => Promise<void>;
  cancelTokenization: (datasetId: string, modelId: string) => Promise<void>;
}

export const useDatasetsStore = create<DatasetsState>()(
  devtools(
    (set, _get) => ({
      // Initial state
      datasets: [],
      tokenizations: {},
      loading: false,
      error: null,

      // Fetch all datasets
      fetchDatasets: async () => {
        set({ loading: true, error: null });
        try {
          const response = await fetch(`${API_BASE_URL}/api/v1/datasets`);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const result = await response.json();
          set({ datasets: result.data || [], loading: false });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch datasets';
          set({ error: errorMessage, loading: false });
        }
      },

      // Download dataset from HuggingFace
      downloadDataset: async (repoId: string, accessToken?: string, split?: string, config?: string) => {
        console.log('[Store] downloadDataset called for repo:', repoId);
        set({ loading: true, error: null });
        try {
          const body: Record<string, string | undefined> = {
            repo_id: repoId,
            access_token: accessToken,
          };

          if (split) body.split = split;
          if (config) body.config = config;

          console.log('[Store] Initiating download request...');
          const response = await fetch(`${API_BASE_URL}/api/v1/datasets/download`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to download dataset');
          }

          const newDataset = await response.json();
          console.log('[Store] Download initiated, dataset ID:', newDataset.id, 'status:', newDataset.status);

          // Ensure status is set properly as a string enum value
          if (newDataset.status) {
            newDataset.status = newDataset.status.toLowerCase();
          }

          // Proactively subscribe to progress updates BEFORE adding to store
          // This ensures we catch early progress events
          if (subscribeToDatasetCallback && newDataset.id) {
            console.log('[Store] Proactively subscribing to dataset:', newDataset.id, 'with status:', newDataset.status);
            subscribeToDatasetCallback(newDataset.id);
          }

          // Add dataset to store
          set((state) => ({
            datasets: [...state.datasets, newDataset],
            loading: false,
          }));

          // Start polling using shared utility
          startPolling<Dataset>({
            fetchStatus: () => getDataset(newDataset.id),
            onUpdate: (updatedDataset) => {
              // Update the dataset in the store
              set((state) => ({
                datasets: state.datasets.map((d) =>
                  d.id === updatedDataset.id
                    ? { ...d, ...updatedDataset, status: updatedDataset.status.toLowerCase() as any }
                    : d
                ),
              }));
            },
            onComplete: (finalDataset) => {
              console.log('[Store] Dataset polling complete:', finalDataset.id, 'final status:', finalDataset.status);
            },
            onError: (error) => {
              console.error('[Store] Dataset polling error:', error);
              set({ error });
            },
            isTerminal: (dataset) => {
              const status = dataset.status.toLowerCase();
              return status !== 'downloading' && status !== 'processing';
            },
            interval: 500,
            maxPolls: 1200,  // 10 minutes at 500ms intervals (increased from 50 to accommodate large dataset downloads)
            resourceId: newDataset.id,
            resourceType: 'dataset',
          });

          return newDataset;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to download dataset';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Delete dataset
      deleteDataset: async (id: string) => {
        set({ loading: true, error: null });
        try {
          const response = await fetch(`${API_BASE_URL}/api/v1/datasets/${id}`, {
            method: 'DELETE',
          });

          if (!response.ok) {
            throw new Error('Failed to delete dataset');
          }

          set((state) => ({
            datasets: state.datasets.filter((d) => d.id !== id),
            loading: false,
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to delete dataset';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Cancel dataset download or tokenization
      cancelDownload: async (id: string) => {
        console.log('[DatasetsStore] cancelDownload called for dataset:', id);
        set({ loading: true, error: null });
        try {
          await cancelDatasetDownload(id);
          console.log('[DatasetsStore] Download/processing cancelled successfully');

          // Update dataset status to error with cancellation message
          // The backend sets status to ERROR with "Cancelled by user"
          set((state) => ({
            datasets: state.datasets.map((d) =>
              d.id === id
                ? {
                    ...d,
                    status: DatasetStatus.ERROR,
                    error_message: 'Cancelled by user',
                    progress: 0,
                  }
                : d
            ),
            loading: false,
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to cancel download';
          console.error('[DatasetsStore] Cancel error:', errorMessage);
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Update dataset progress (called by WebSocket)
      updateDatasetProgress: (id: string, progress: number) => {
        set((state) => ({
          datasets: state.datasets.map((dataset) =>
            dataset.id === id
              ? { ...dataset, progress }
              : dataset
          ),
        }));
      },

      // Update dataset status (called by WebSocket)
      updateDatasetStatus: (id: string, status: DatasetStatus, errorMessage?: string) => {
        set((state) => ({
          datasets: state.datasets.map((dataset) =>
            dataset.id === id
              ? {
                  ...dataset,
                  status,
                  error_message: errorMessage,
                  progress: status === DatasetStatus.READY ? 100 : dataset.progress
                }
              : dataset
          ),
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

      // Fetch tokenizations for a dataset
      fetchTokenizations: async (datasetId: string) => {
        try {
          const response = await fetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/tokenizations`);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const result: DatasetTokenizationListResponse = await response.json();
          set((state) => ({
            tokenizations: {
              ...state.tokenizations,
              [datasetId]: result.data || [],
            },
          }));
        } catch (error) {
          console.error('[Store] Failed to fetch tokenizations:', error);
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch tokenizations';
          set({ error: errorMessage });
        }
      },

      // Create a new tokenization
      createTokenization: async (datasetId: string, modelId: string, params: any) => {
        set({ loading: true, error: null });
        try {
          const response = await fetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/tokenize`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              model_id: modelId,
              ...params,
            }),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to create tokenization');
          }

          // Refresh tokenizations list
          await _get().fetchTokenizations(datasetId);
          set({ loading: false });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to create tokenization';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Delete a tokenization
      deleteTokenization: async (datasetId: string, modelId: string) => {
        set({ loading: true, error: null });
        try {
          const response = await fetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/tokenizations/${modelId}`, {
            method: 'DELETE',
          });

          if (!response.ok) {
            throw new Error('Failed to delete tokenization');
          }

          // Update local state
          set((state) => ({
            tokenizations: {
              ...state.tokenizations,
              [datasetId]: (state.tokenizations[datasetId] || []).filter((t) => t.model_id !== modelId),
            },
            loading: false,
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to delete tokenization';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Cancel a tokenization
      cancelTokenization: async (datasetId: string, modelId: string) => {
        set({ loading: true, error: null });
        try {
          const response = await fetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/tokenizations/${modelId}/cancel`, {
            method: 'POST',
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to cancel tokenization');
          }

          // Refresh tokenizations list to get updated status
          await _get().fetchTokenizations(datasetId);
          set({ loading: false });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to cancel tokenization';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },
    }),
    {
      name: 'DatasetsStore',
    }
  )
);
