/**
 * Unit tests for modelsStore.
 *
 * These tests verify the Zustand store's state management and API integration.
 * Fetch calls are mocked to test store behavior without requiring a live backend.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useModelsStore, setModelSubscriptionCallback } from './modelsStore';
import { Model, ModelStatus, QuantizationFormat } from '../types/model';

// Mock the API_BASE_URL
vi.mock('../config/api', () => ({
  API_BASE_URL: 'http://localhost:8000',
}));

describe('modelsStore', () => {
  // Reset store state before each test
  beforeEach(() => {
    useModelsStore.setState({
      models: [],
      loading: false,
      error: null,
    });
    // Clear all fetch mocks
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('fetchModels', () => {
    it('should fetch models successfully', async () => {
      const mockModels: Model[] = [
        {
          id: 'm_test123',
          name: 'TinyLlama-1.1B',
          repo_id: 'TinyLlama/TinyLlama-1.1B',
          architecture: 'llama',
          params_count: 1100000000,
          quantization: QuantizationFormat.Q4,
          status: ModelStatus.READY,
          progress: 100,
          created_at: '2025-10-12T00:00:00Z',
          updated_at: '2025-10-12T00:00:00Z',
        },
      ];

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: mockModels }),
      });

      const { fetchModels } = useModelsStore.getState();
      await fetchModels();

      const state = useModelsStore.getState();
      expect(state.models).toEqual(mockModels);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
      expect(global.fetch).toHaveBeenCalledWith('http://localhost:8000/api/v1/models');
    });

    it('should handle fetch error', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 500,
      });

      const { fetchModels } = useModelsStore.getState();
      await fetchModels();

      const state = useModelsStore.getState();
      expect(state.models).toEqual([]);
      expect(state.loading).toBe(false);
      expect(state.error).toBe('HTTP error! status: 500');
    });

    it('should handle network error', async () => {
      global.fetch = vi.fn().mockRejectedValueOnce(new Error('Network error'));

      const { fetchModels } = useModelsStore.getState();
      await fetchModels();

      const state = useModelsStore.getState();
      expect(state.models).toEqual([]);
      expect(state.loading).toBe(false);
      expect(state.error).toBe('Network error');
    });
  });

  describe('downloadModel', () => {
    it('should download model successfully', async () => {
      const mockModel: Model = {
        id: 'm_newmodel',
        name: 'GPT2',
        repo_id: 'gpt2',
        architecture: 'gpt2',
        params_count: 124000000,
        quantization: QuantizationFormat.Q8,
        status: ModelStatus.DOWNLOADING,
        progress: 0,
        created_at: '2025-10-12T00:00:00Z',
        updated_at: '2025-10-12T00:00:00Z',
      };

      // Mock the download POST request
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockModel,
      });

      // Mock subscription callback
      const mockSubscribe = vi.fn();
      setModelSubscriptionCallback(mockSubscribe);

      const { downloadModel } = useModelsStore.getState();
      await downloadModel('gpt2', QuantizationFormat.Q8);

      const state = useModelsStore.getState();
      expect(state.models).toHaveLength(1);
      expect(state.models[0].id).toBe('m_newmodel');
      expect(state.models[0].status).toBe('downloading');
      expect(state.loading).toBe(false);
      expect(mockSubscribe).toHaveBeenCalledWith('m_newmodel', 'progress');
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models/download',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            repo_id: 'gpt2',
            quantization: 'Q8',
            trust_remote_code: false,
          }),
        })
      );
    });

    it('should download model with access token', async () => {
      const mockModel: Model = {
        id: 'm_gated',
        name: 'GatedModel',
        repo_id: 'org/gated-model',
        architecture: 'llama',
        params_count: 7000000000,
        quantization: QuantizationFormat.Q4,
        status: ModelStatus.DOWNLOADING,
        progress: 0,
        created_at: '2025-10-12T00:00:00Z',
        updated_at: '2025-10-12T00:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockModel,
      });

      const { downloadModel } = useModelsStore.getState();
      await downloadModel('org/gated-model', QuantizationFormat.Q4, 'test_token_123');

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models/download',
        expect.objectContaining({
          body: JSON.stringify({
            repo_id: 'org/gated-model',
            quantization: 'Q4',
            trust_remote_code: false,
            access_token: 'test_token_123',
          }),
        })
      );
    });

    it('should handle download error', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'Invalid repo_id' }),
      });

      const { downloadModel } = useModelsStore.getState();

      await expect(downloadModel('invalid/repo', QuantizationFormat.Q4)).rejects.toThrow(
        'Invalid repo_id'
      );

      const state = useModelsStore.getState();
      expect(state.error).toBe('Invalid repo_id');
      expect(state.loading).toBe(false);
    });
  });

  describe('deleteModel', () => {
    it('should delete model successfully', async () => {
      // Setup initial state with a model
      useModelsStore.setState({
        models: [
          {
            id: 'm_delete_me',
            name: 'ModelToDelete',
            architecture: 'gpt2',
            params_count: 124000000,
            quantization: QuantizationFormat.Q4,
            status: ModelStatus.READY,
            created_at: '2025-10-12T00:00:00Z',
            updated_at: '2025-10-12T00:00:00Z',
          },
        ],
      });

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        status: 204,
      });

      const { deleteModel } = useModelsStore.getState();
      await deleteModel('m_delete_me');

      const state = useModelsStore.getState();
      expect(state.models).toHaveLength(0);
      expect(state.loading).toBe(false);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models/m_delete_me',
        { method: 'DELETE' }
      );
    });

    it('should handle delete error', async () => {
      useModelsStore.setState({
        models: [
          {
            id: 'm_cantdelete',
            name: 'ProtectedModel',
            architecture: 'llama',
            params_count: 1100000000,
            quantization: QuantizationFormat.Q8,
            status: ModelStatus.READY,
            created_at: '2025-10-12T00:00:00Z',
            updated_at: '2025-10-12T00:00:00Z',
          },
        ],
      });

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 403,
      });

      const { deleteModel } = useModelsStore.getState();
      await expect(deleteModel('m_cantdelete')).rejects.toThrow();

      const state = useModelsStore.getState();
      expect(state.models).toHaveLength(1); // Model still present
      expect(state.error).toBe('Failed to delete model');
    });
  });

  describe('cancelDownload', () => {
    it('should cancel download successfully', async () => {
      // Setup initial state with a downloading model
      useModelsStore.setState({
        models: [
          {
            id: 'm_cancel_me',
            name: 'ModelToCancel',
            repo_id: 'org/model-to-cancel',
            architecture: 'llama',
            params_count: 1100000000,
            quantization: QuantizationFormat.Q4,
            status: ModelStatus.DOWNLOADING,
            progress: 45,
            created_at: '2025-10-13T00:00:00Z',
            updated_at: '2025-10-13T00:00:00Z',
          },
        ],
      });

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'cancelled',
          message: 'Download cancelled successfully',
        }),
      });

      const { cancelDownload } = useModelsStore.getState();
      await cancelDownload('m_cancel_me');

      const state = useModelsStore.getState();
      expect(state.models).toHaveLength(0); // Model removed from store
      expect(state.loading).toBe(false);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models/m_cancel_me/cancel',
        { method: 'DELETE' }
      );
    });

    it('should handle cancel error for non-cancellable model', async () => {
      useModelsStore.setState({
        models: [
          {
            id: 'm_ready_model',
            name: 'ReadyModel',
            architecture: 'gpt2',
            params_count: 124000000,
            quantization: QuantizationFormat.Q8,
            status: ModelStatus.READY,
            progress: 100,
            created_at: '2025-10-13T00:00:00Z',
            updated_at: '2025-10-13T00:00:00Z',
          },
        ],
      });

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({
          detail: "Model 'm_ready_model' cannot be cancelled (status: ready)",
        }),
      });

      const { cancelDownload } = useModelsStore.getState();
      await expect(cancelDownload('m_ready_model')).rejects.toThrow(
        "Model 'm_ready_model' cannot be cancelled (status: ready)"
      );

      const state = useModelsStore.getState();
      expect(state.models).toHaveLength(1); // Model still present
      expect(state.error).toBe("Model 'm_ready_model' cannot be cancelled (status: ready)");
    });

    it('should handle network error during cancel', async () => {
      useModelsStore.setState({
        models: [
          {
            id: 'm_network_fail',
            name: 'NetworkFailModel',
            architecture: 'llama',
            params_count: 7000000000,
            quantization: QuantizationFormat.Q4,
            status: ModelStatus.LOADING,
            progress: 75,
            created_at: '2025-10-13T00:00:00Z',
            updated_at: '2025-10-13T00:00:00Z',
          },
        ],
      });

      global.fetch = vi.fn().mockRejectedValueOnce(new Error('Network error'));

      const { cancelDownload } = useModelsStore.getState();
      await expect(cancelDownload('m_network_fail')).rejects.toThrow('Network error');

      const state = useModelsStore.getState();
      expect(state.models).toHaveLength(1); // Model still present on error
      expect(state.error).toBe('Network error');
    });
  });

  describe('extractActivations', () => {
    it('should initiate extraction successfully', async () => {
      const mockResult = {
        job_id: 'job_abc123',
        status: 'queued',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockResult,
      });

      const mockSubscribe = vi.fn();
      setModelSubscriptionCallback(mockSubscribe);

      const { extractActivations } = useModelsStore.getState();
      await extractActivations(
        'm_model1',
        'd_dataset1',
        [0, 5, 10],
        ['residual', 'mlp'],
        1000,
        32
      );

      const state = useModelsStore.getState();
      expect(state.loading).toBe(false);
      expect(mockSubscribe).toHaveBeenCalledWith('m_model1', 'extraction');
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models/m_model1/extract-activations',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            dataset_id: 'd_dataset1',
            layer_indices: [0, 5, 10],
            hook_types: ['residual', 'mlp'],
            max_samples: 1000,
            batch_size: 32,
          }),
        })
      );
    });

    it('should handle extraction error', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Model not found' }),
      });

      const { extractActivations } = useModelsStore.getState();
      await expect(
        extractActivations('m_nonexistent', 'd_dataset1', [0], ['residual'], 100)
      ).rejects.toThrow('Model not found');

      const state = useModelsStore.getState();
      expect(state.error).toBe('Model not found');
    });
  });

  describe('updateModelProgress', () => {
    it('should update model progress', () => {
      useModelsStore.setState({
        models: [
          {
            id: 'm_progress',
            name: 'ProgressModel',
            architecture: 'gpt2',
            params_count: 124000000,
            quantization: QuantizationFormat.Q4,
            status: ModelStatus.DOWNLOADING,
            progress: 0,
            created_at: '2025-10-12T00:00:00Z',
            updated_at: '2025-10-12T00:00:00Z',
          },
        ],
      });

      const { updateModelProgress } = useModelsStore.getState();
      updateModelProgress('m_progress', 50);

      const state = useModelsStore.getState();
      expect(state.models[0].progress).toBe(50);
      expect(state.models[0].status).toBe(ModelStatus.DOWNLOADING);
    });

    it('should not update progress for non-existent model', () => {
      useModelsStore.setState({
        models: [
          {
            id: 'm_other',
            name: 'OtherModel',
            architecture: 'gpt2',
            params_count: 124000000,
            quantization: QuantizationFormat.Q4,
            status: ModelStatus.READY,
            progress: 100,
            created_at: '2025-10-12T00:00:00Z',
            updated_at: '2025-10-12T00:00:00Z',
          },
        ],
      });

      const { updateModelProgress } = useModelsStore.getState();
      updateModelProgress('m_nonexistent', 75);

      const state = useModelsStore.getState();
      expect(state.models[0].progress).toBe(100); // Unchanged
    });
  });

  describe('updateModelStatus', () => {
    it('should update model status', () => {
      useModelsStore.setState({
        models: [
          {
            id: 'm_status',
            name: 'StatusModel',
            architecture: 'llama',
            params_count: 1100000000,
            quantization: QuantizationFormat.Q8,
            status: ModelStatus.DOWNLOADING,
            progress: 50,
            created_at: '2025-10-12T00:00:00Z',
            updated_at: '2025-10-12T00:00:00Z',
          },
        ],
      });

      const { updateModelStatus } = useModelsStore.getState();
      updateModelStatus('m_status', ModelStatus.LOADING);

      const state = useModelsStore.getState();
      expect(state.models[0].status).toBe(ModelStatus.LOADING);
      expect(state.models[0].progress).toBe(50);
    });

    it('should set progress to 100 when status is READY', () => {
      useModelsStore.setState({
        models: [
          {
            id: 'm_ready',
            name: 'ReadyModel',
            architecture: 'gpt2',
            params_count: 124000000,
            quantization: QuantizationFormat.Q4,
            status: ModelStatus.QUANTIZING,
            progress: 95,
            created_at: '2025-10-12T00:00:00Z',
            updated_at: '2025-10-12T00:00:00Z',
          },
        ],
      });

      const { updateModelStatus } = useModelsStore.getState();
      updateModelStatus('m_ready', ModelStatus.READY);

      const state = useModelsStore.getState();
      expect(state.models[0].status).toBe(ModelStatus.READY);
      expect(state.models[0].progress).toBe(100);
    });

    it('should set error message when status is ERROR', () => {
      useModelsStore.setState({
        models: [
          {
            id: 'm_error',
            name: 'ErrorModel',
            architecture: 'llama',
            params_count: 7000000000,
            quantization: QuantizationFormat.Q2,
            status: ModelStatus.DOWNLOADING,
            progress: 25,
            created_at: '2025-10-12T00:00:00Z',
            updated_at: '2025-10-12T00:00:00Z',
          },
        ],
      });

      const { updateModelStatus } = useModelsStore.getState();
      updateModelStatus('m_error', ModelStatus.ERROR, 'Out of memory');

      const state = useModelsStore.getState();
      expect(state.models[0].status).toBe(ModelStatus.ERROR);
      expect(state.models[0].error_message).toBe('Out of memory');
    });
  });

  describe('error management', () => {
    it('should set error message', () => {
      const { setError } = useModelsStore.getState();
      setError('Test error message');

      const state = useModelsStore.getState();
      expect(state.error).toBe('Test error message');
    });

    it('should clear error message', () => {
      useModelsStore.setState({ error: 'Some error' });

      const { clearError } = useModelsStore.getState();
      clearError();

      const state = useModelsStore.getState();
      expect(state.error).toBeNull();
    });
  });

  describe('WebSocket subscription callback', () => {
    it('should register subscription callback', async () => {
      const mockCallback = vi.fn();
      setModelSubscriptionCallback(mockCallback);

      // Trigger a download to test the callback
      const mockModel: Model = {
        id: 'm_callback_test',
        name: 'CallbackTest',
        architecture: 'gpt2',
        params_count: 124000000,
        quantization: QuantizationFormat.Q4,
        status: ModelStatus.DOWNLOADING,
        progress: 0,
        created_at: '2025-10-12T00:00:00Z',
        updated_at: '2025-10-12T00:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockModel,
      });

      const { downloadModel } = useModelsStore.getState();
      await downloadModel('gpt2', QuantizationFormat.Q4);

      // Callback should have been invoked after download completes
      expect(mockCallback).toHaveBeenCalledWith('m_callback_test', 'progress');
    });
  });
});
