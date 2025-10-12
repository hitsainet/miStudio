/**
 * Zustand store for model management.
 *
 * This store manages the global state for models, including:
 * - List of models
 * - Loading states
 * - Error handling
 * - CRUD operations (download, delete)
 * - Real-time progress updates via WebSocket
 * - Activation extraction management
 *
 * Connects to REAL backend API at /api/v1/models
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { Model, ModelStatus, QuantizationFormat } from '../types/model';
import { API_BASE_URL } from '../config/api';
import { getModel } from '../api/models';
import { startPolling } from '../utils/polling';

// Callback for subscribing to model progress (set by WebSocket context)
let subscribeToModelCallback: ((modelId: string, channel: 'progress' | 'extraction') => void) | null = null;

export function setModelSubscriptionCallback(callback: (modelId: string, channel: 'progress' | 'extraction') => void) {
  subscribeToModelCallback = callback;
}

interface ModelsState {
  // State
  models: Model[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchModels: () => Promise<void>;
  downloadModel: (repoId: string, quantization: QuantizationFormat, accessToken?: string, trustRemoteCode?: boolean) => Promise<void>;
  deleteModel: (id: string) => Promise<void>;
  cancelDownload: (id: string) => Promise<void>;
  extractActivations: (modelId: string, datasetId: string, layerIndices: number[], hookTypes: string[], maxSamples: number, batchSize?: number) => Promise<void>;
  updateModelProgress: (id: string, progress: number) => void;
  updateModelStatus: (id: string, status: ModelStatus, errorMessage?: string) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
}

export const useModelsStore = create<ModelsState>()(
  devtools(
    (set, _get) => ({
      // Initial state
      models: [],
      loading: false,
      error: null,

      // Fetch all models
      fetchModels: async () => {
        set({ loading: true, error: null });
        try {
          const response = await fetch(`${API_BASE_URL}/api/v1/models`);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const result = await response.json();
          set({ models: result.data || [], loading: false });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch models';
          set({ error: errorMessage, loading: false });
        }
      },

      // Download model from HuggingFace
      downloadModel: async (repoId: string, quantization: QuantizationFormat, accessToken?: string, trustRemoteCode?: boolean) => {
        console.log('[ModelsStore] downloadModel called for repo:', repoId, 'quantization:', quantization, 'trustRemoteCode:', trustRemoteCode);
        set({ loading: true, error: null });
        try {
          const body: Record<string, any> = {
            repo_id: repoId,
            quantization,
            trust_remote_code: trustRemoteCode || false,
          };

          if (accessToken) {
            body.access_token = accessToken;
          }

          console.log('[ModelsStore] Initiating download request...');
          const response = await fetch(`${API_BASE_URL}/api/v1/models/download`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to download model');
          }

          const newModel = await response.json();
          console.log('[ModelsStore] Download initiated, model ID:', newModel.id, 'status:', newModel.status);

          // Ensure status is set properly as a string enum value
          if (newModel.status) {
            newModel.status = newModel.status.toLowerCase();
          }

          // Proactively subscribe to progress updates BEFORE adding to store
          // This ensures we catch early progress events
          if (subscribeToModelCallback && newModel.id) {
            console.log('[ModelsStore] Proactively subscribing to model progress:', newModel.id, 'with status:', newModel.status);
            subscribeToModelCallback(newModel.id, 'progress');
          }

          // Add model to store
          set((state) => ({
            models: [...state.models, newModel],
            loading: false,
          }));

          // Start polling using shared utility
          startPolling<Model>({
            fetchStatus: () => getModel(newModel.id),
            onUpdate: (updatedModel) => {
              // Update the model in the store
              set((state) => ({
                models: state.models.map((m) =>
                  m.id === updatedModel.id
                    ? { ...m, ...updatedModel, status: updatedModel.status.toLowerCase() as any }
                    : m
                ),
              }));
            },
            onComplete: (finalModel) => {
              console.log('[ModelsStore] Model polling complete:', finalModel.id, 'final status:', finalModel.status);
            },
            onError: (error) => {
              console.error('[ModelsStore] Model polling error:', error);
              set({ error });
            },
            isTerminal: (model) => {
              const status = model.status.toLowerCase();
              return status !== 'downloading' && status !== 'loading' && status !== 'quantizing';
            },
            interval: 500,
            maxPolls: 100,
            resourceId: newModel.id,
            resourceType: 'model',
          });

          return newModel;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to download model';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Delete model
      deleteModel: async (id: string) => {
        set({ loading: true, error: null });
        try {
          const response = await fetch(`${API_BASE_URL}/api/v1/models/${id}`, {
            method: 'DELETE',
          });

          if (!response.ok) {
            throw new Error('Failed to delete model');
          }

          set((state) => ({
            models: state.models.filter((m) => m.id !== id),
            loading: false,
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to delete model';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Cancel model download
      cancelDownload: async (id: string) => {
        set({ loading: true, error: null });
        try {
          const response = await fetch(`${API_BASE_URL}/api/v1/models/${id}/cancel`, {
            method: 'DELETE',
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to cancel download');
          }

          // Remove cancelled model from list
          set((state) => ({
            models: state.models.filter((m) => m.id !== id),
            loading: false,
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to cancel download';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Extract activations from model
      extractActivations: async (
        modelId: string,
        datasetId: string,
        layerIndices: number[],
        hookTypes: string[],
        maxSamples: number,
        batchSize?: number
      ) => {
        console.log('[ModelsStore] extractActivations called for model:', modelId);
        set({ loading: true, error: null });
        try {
          const body = {
            dataset_id: datasetId,
            layer_indices: layerIndices,
            hook_types: hookTypes,
            max_samples: maxSamples,
            ...(batchSize && { batch_size: batchSize }),
          };

          console.log('[ModelsStore] Initiating extraction request...');
          const response = await fetch(`${API_BASE_URL}/api/v1/models/${modelId}/extract-activations`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to extract activations');
          }

          const result = await response.json();
          console.log('[ModelsStore] Extraction initiated, job_id:', result.job_id);

          // Subscribe to extraction progress updates
          if (subscribeToModelCallback) {
            console.log('[ModelsStore] Subscribing to extraction progress for model:', modelId);
            subscribeToModelCallback(modelId, 'extraction');
          }

          set({ loading: false });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to extract activations';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Update model progress (called by WebSocket)
      updateModelProgress: (id: string, progress: number) => {
        set((state) => ({
          models: state.models.map((model) =>
            model.id === id
              ? { ...model, progress }
              : model
          ),
        }));
      },

      // Update model status (called by WebSocket)
      updateModelStatus: (id: string, status: ModelStatus, errorMessage?: string) => {
        set((state) => ({
          models: state.models.map((model) =>
            model.id === id
              ? {
                  ...model,
                  status,
                  error_message: errorMessage,
                  progress: status === ModelStatus.READY ? 100 : model.progress
                }
              : model
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
      name: 'ModelsStore',
    }
  )
);
