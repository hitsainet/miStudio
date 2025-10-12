/**
 * Unit tests for models API client.
 *
 * These tests verify all API functions make correct HTTP requests
 * and handle responses/errors properly. Fetch calls are mocked.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  getModels,
  getModel,
  downloadModel,
  deleteModel,
  getModelArchitecture,
  extractActivations,
  updateModel,
  getTaskStatus,
} from './models';
import { QuantizationFormat, ModelStatus } from '../types/model';

// Mock the API_BASE_URL
vi.mock('../config/api', () => ({
  API_BASE_URL: 'http://localhost:8000',
}));

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};
global.localStorage = localStorageMock as any;

describe('models API client', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
    localStorageMock.getItem.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('getModels', () => {
    it('should fetch models without parameters', async () => {
      const mockResponse = {
        data: [
          {
            id: 'm_test1',
            name: 'Test Model',
            architecture: 'gpt2',
            params_count: 124000000,
            quantization: QuantizationFormat.Q4,
            status: ModelStatus.READY,
            created_at: '2025-10-12T00:00:00Z',
            updated_at: '2025-10-12T00:00:00Z',
          },
        ],
        pagination: { page: 1, total: 1, has_next: false },
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await getModels();

      expect(result).toEqual(mockResponse);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models',
        expect.objectContaining({
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
        })
      );
    });

    it('should fetch models with query parameters', async () => {
      const mockResponse = {
        data: [],
        pagination: { page: 2, total: 0, has_next: false },
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      await getModels({
        skip: 10,
        limit: 5,
        search: 'llama',
        architecture: 'llama',
        quantization: 'Q4',
        status: 'ready',
        sort_by: 'created_at',
        order: 'desc',
      });

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models?skip=10&limit=5&search=llama&architecture=llama&quantization=Q4&status=ready&sort_by=created_at&order=desc',
        expect.any(Object)
      );
    });

    it('should include auth token if present', async () => {
      localStorageMock.getItem.mockReturnValueOnce('test_auth_token');

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [] }),
      });

      await getModels();

      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer test_auth_token',
          }),
        })
      );
    });
  });

  describe('getModel', () => {
    it('should fetch a single model by ID', async () => {
      const mockModel = {
        id: 'm_single',
        name: 'Single Model',
        architecture: 'llama',
        params_count: 1100000000,
        quantization: QuantizationFormat.Q8,
        status: ModelStatus.READY,
        created_at: '2025-10-12T00:00:00Z',
        updated_at: '2025-10-12T00:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockModel,
      });

      const result = await getModel('m_single');

      expect(result).toEqual(mockModel);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models/m_single',
        expect.any(Object)
      );
    });

    it('should throw error if model not found', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Model not found' }),
      });

      await expect(getModel('m_nonexistent')).rejects.toThrow('Model not found');
    });
  });

  describe('downloadModel', () => {
    it('should download model with repo_id and quantization', async () => {
      const mockModel = {
        id: 'm_download',
        name: 'Downloaded Model',
        repo_id: 'gpt2',
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

      const result = await downloadModel({
        repo_id: 'gpt2',
        quantization: QuantizationFormat.Q4,
      });

      expect(result).toEqual(mockModel);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models/download',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            repo_id: 'gpt2',
            quantization: 'Q4',
          }),
        })
      );
    });

    it('should download model with access token', async () => {
      const mockModel = {
        id: 'm_gated',
        name: 'Gated Model',
        repo_id: 'org/gated-model',
        architecture: 'llama',
        params_count: 7000000000,
        quantization: QuantizationFormat.Q4,
        status: ModelStatus.DOWNLOADING,
        created_at: '2025-10-12T00:00:00Z',
        updated_at: '2025-10-12T00:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockModel,
      });

      await downloadModel({
        repo_id: 'org/gated-model',
        quantization: QuantizationFormat.Q4,
        access_token: 'hf_token_123',
      });

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models/download',
        expect.objectContaining({
          body: JSON.stringify({
            repo_id: 'org/gated-model',
            quantization: 'Q4',
            access_token: 'hf_token_123',
          }),
        })
      );
    });
  });

  describe('deleteModel', () => {
    it('should delete a model successfully', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        status: 204,
      });

      const result = await deleteModel('m_delete');

      expect(result).toBeUndefined();
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models/m_delete',
        expect.objectContaining({
          method: 'DELETE',
        })
      );
    });

    it('should throw error if delete fails', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 403,
        json: async () => ({ detail: 'Cannot delete model in use' }),
      });

      await expect(deleteModel('m_inuse')).rejects.toThrow('Cannot delete model in use');
    });
  });

  describe('getModelArchitecture', () => {
    it('should fetch model architecture details', async () => {
      const mockArchitecture = {
        architecture: 'LlamaForCausalLM',
        hidden_size: 2048,
        num_layers: 22,
        num_attention_heads: 32,
        intermediate_size: 5632,
        vocab_size: 32000,
        layers: [
          { index: 0, name: 'model.layers.0', type: 'TransformerBlock' },
          { index: 1, name: 'model.layers.1', type: 'TransformerBlock' },
        ],
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockArchitecture,
      });

      const result = await getModelArchitecture('m_arch');

      expect(result).toEqual(mockArchitecture);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models/m_arch/architecture',
        expect.any(Object)
      );
    });
  });

  describe('extractActivations', () => {
    it('should initiate activation extraction', async () => {
      const mockResult = {
        extraction_id: 'ext_abc123',
        output_path: '/data/extractions/ext_abc123',
        num_samples: 1000,
        saved_files: ['layer_0_residual.npy', 'layer_5_mlp.npy'],
        statistics: {
          'layer_0_residual': {
            shape: [1000, 512, 2048],
            mean_magnitude: 0.5,
            max_activation: 12.3,
            min_activation: -8.7,
            std_activation: 2.1,
            sparsity_percent: 35.2,
            size_mb: 4096,
          },
        },
        metadata_path: '/data/extractions/ext_abc123/metadata.json',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockResult,
      });

      const result = await extractActivations('m_extract', {
        dataset_id: 'd_dataset1',
        layer_indices: [0, 5, 10],
        hook_types: ['residual', 'mlp'],
        max_samples: 1000,
        batch_size: 32,
        top_k_examples: 10,
      });

      expect(result).toEqual(mockResult);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models/m_extract/extract-activations',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            dataset_id: 'd_dataset1',
            layer_indices: [0, 5, 10],
            hook_types: ['residual', 'mlp'],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
          }),
        })
      );
    });

    it('should handle extraction error', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'Invalid layer indices' }),
      });

      await expect(
        extractActivations('m_bad', {
          dataset_id: 'd_dataset1',
          layer_indices: [-1, 999],
          hook_types: ['residual'],
          max_samples: 100,
        })
      ).rejects.toThrow('Invalid layer indices');
    });
  });

  describe('updateModel', () => {
    it('should update model metadata', async () => {
      const mockUpdatedModel = {
        id: 'm_update',
        name: 'Updated Model Name',
        architecture: 'gpt2',
        params_count: 124000000,
        quantization: QuantizationFormat.Q4,
        status: ModelStatus.READY,
        created_at: '2025-10-12T00:00:00Z',
        updated_at: '2025-10-12T01:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockUpdatedModel,
      });

      const result = await updateModel('m_update', {
        name: 'Updated Model Name',
      });

      expect(result).toEqual(mockUpdatedModel);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models/m_update',
        expect.objectContaining({
          method: 'PATCH',
          body: JSON.stringify({ name: 'Updated Model Name' }),
        })
      );
    });
  });

  describe('getTaskStatus', () => {
    it('should fetch Celery task status', async () => {
      const mockTaskStatus = {
        task_id: 'task_xyz789',
        status: 'SUCCESS',
        result: {
          model_id: 'm_completed',
          status: 'ready',
        },
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockTaskStatus,
      });

      const result = await getTaskStatus('task_xyz789');

      expect(result).toEqual(mockTaskStatus);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/models/tasks/task_xyz789',
        expect.any(Object)
      );
    });

    it('should fetch task status with error', async () => {
      const mockTaskStatus = {
        task_id: 'task_failed',
        status: 'FAILURE',
        error: 'Out of memory',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockTaskStatus,
      });

      const result = await getTaskStatus('task_failed');

      expect(result.status).toBe('FAILURE');
      expect(result.error).toBe('Out of memory');
    });
  });

  describe('error handling', () => {
    it('should handle network error', async () => {
      global.fetch = vi.fn().mockRejectedValueOnce(new Error('Network error'));

      await expect(getModels()).rejects.toThrow('Network error');
    });

    it('should handle JSON parse error', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => {
          throw new Error('Invalid JSON');
        },
      });

      await expect(getModels()).rejects.toThrow('HTTP error! status: 500');
    });

    it('should handle 204 No Content response', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        status: 204,
      });

      const result = await deleteModel('m_delete');
      expect(result).toBeUndefined();
    });
  });
});
