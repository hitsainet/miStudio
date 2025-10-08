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

interface DatasetsState {
  // State
  datasets: Dataset[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchDatasets: () => Promise<void>;
  downloadDataset: (repoId: string, accessToken?: string) => Promise<void>;
  deleteDataset: (id: string) => Promise<void>;
  updateDatasetProgress: (id: string, progress: number) => void;
  updateDatasetStatus: (id: string, status: DatasetStatus, errorMessage?: string) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
}

export const useDatasetsStore = create<DatasetsState>()(
  devtools(
    (set, get) => ({
      // Initial state
      datasets: [],
      loading: false,
      error: null,

      // Fetch all datasets
      fetchDatasets: async () => {
        set({ loading: true, error: null });
        try {
          const response = await fetch('http://localhost:8000/api/v1/datasets');
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
      downloadDataset: async (repoId: string, accessToken?: string) => {
        set({ loading: true, error: null });
        try {
          const response = await fetch('http://localhost:8000/api/v1/datasets/download', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              repo_id: repoId,
              access_token: accessToken,
            }),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to download dataset');
          }

          const newDataset = await response.json();
          set((state) => ({
            datasets: [...state.datasets, newDataset],
            loading: false,
          }));
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
          const response = await fetch(`http://localhost:8000/api/v1/datasets/${id}`, {
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
