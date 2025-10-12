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
import { Dataset, DatasetStatus } from '../types/dataset';
import { API_BASE_URL } from '../config/api';
import { cancelDatasetDownload } from '../api/datasets';

// Callback for subscribing to dataset progress (set by WebSocket context)
let subscribeToDatasetCallback: ((datasetId: string) => void) | null = null;

export function setDatasetSubscriptionCallback(callback: (datasetId: string) => void) {
  subscribeToDatasetCallback = callback;
}

interface DatasetsState {
  // State
  datasets: Dataset[];
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
}

export const useDatasetsStore = create<DatasetsState>()(
  devtools(
    (set, _get) => ({
      // Initial state
      datasets: [],
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

          console.log('[Store] About to start polling for dataset:', newDataset.id);

          // Start aggressive polling for fast downloads
          // This catches progress updates even for very quick downloads
          const startPolling = () => {
            console.log('[Store] startPolling() function called');
            let pollCount = 0;
            const maxPolls = 50; // Maximum 25 seconds of polling (50 * 500ms)

            const pollInterval = setInterval(async () => {
              pollCount++;

              try {
                const pollResponse = await fetch(`${API_BASE_URL}/api/v1/datasets/${newDataset.id}`);
                if (pollResponse.ok) {
                  const updatedDataset = await pollResponse.json();

                  console.log(`[Store] Poll ${pollCount}: Dataset ${newDataset.id} status=${updatedDataset.status}, progress=${updatedDataset.progress}`);

                  // Update the dataset in the store
                  set((state) => ({
                    datasets: state.datasets.map((d) =>
                      d.id === updatedDataset.id ? { ...d, ...updatedDataset, status: updatedDataset.status.toLowerCase() } : d
                    ),
                  }));

                  // Stop polling if no longer downloading/processing
                  if (updatedDataset.status !== 'downloading' && updatedDataset.status !== 'processing') {
                    console.log(`[Store] Stopping poll for dataset ${newDataset.id} - final status: ${updatedDataset.status}`);
                    clearInterval(pollInterval);
                  }
                }
              } catch (error) {
                console.error('[Store] Polling error:', error);
              }

              // Stop after max polls
              if (pollCount >= maxPolls) {
                console.log(`[Store] Stopping poll for dataset ${newDataset.id} - max polls reached`);
                clearInterval(pollInterval);
              }
            }, 500); // Poll every 500ms for fast updates
          };

          // Start polling immediately
          startPolling();

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
    }),
    {
      name: 'DatasetsStore',
    }
  )
);
