/**
 * Unit tests for datasets API client.
 *
 * This module tests all API client functions with mocked fetch responses,
 * including query parameter construction, error handling, and authentication.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import {
  getDatasets,
  getDataset,
  downloadDataset,
  deleteDataset,
  getDatasetSamples,
  getDatasetStatistics,
  tokenizeDataset,
} from './datasets';
import { DatasetStatus } from '../types/dataset';

// Mock fetch globally
global.fetch = vi.fn();

// Mock localStorage
const mockLocalStorage = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
  length: 0,
  key: vi.fn(),
};
Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
  writable: true,
});

describe('datasets API client', () => {
  beforeEach(() => {
    // Reset mocks before each test
    vi.clearAllMocks();
    mockLocalStorage.getItem.mockReturnValue(null);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('getDatasets', () => {
    it('should fetch datasets without parameters', async () => {
      const mockResponse = {
        data: [
          {
            id: '1',
            name: 'test-dataset',
            status: DatasetStatus.READY,
            progress: 100,
            created_at: new Date().toISOString(),
          },
        ],
        total: 1,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await getDatasets();

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/datasets'),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('should fetch datasets with pagination parameters', async () => {
      const mockResponse = { data: [], total: 0 };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      await getDatasets({ skip: 10, limit: 20 });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('skip=10'),
        expect.anything()
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('limit=20'),
        expect.anything()
      );
    });

    it('should fetch datasets with search parameter', async () => {
      const mockResponse = { data: [], total: 0 };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      await getDatasets({ search: 'test' });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('search=test'),
        expect.anything()
      );
    });

    it('should fetch datasets with status filter', async () => {
      const mockResponse = { data: [], total: 0 };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      await getDatasets({ status: 'ready' });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('status=ready'),
        expect.anything()
      );
    });

    it('should fetch datasets with sorting parameters', async () => {
      const mockResponse = { data: [], total: 0 };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      await getDatasets({ sort_by: 'created_at', order: 'desc' });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('sort_by=created_at'),
        expect.anything()
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('order=desc'),
        expect.anything()
      );
    });

    it('should include auth token when present', async () => {
      mockLocalStorage.getItem.mockReturnValue('test_token');

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [], total: 0 }),
      });

      await getDatasets();

      expect(global.fetch).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer test_token',
          }),
        })
      );
    });

    it('should handle API errors', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ detail: 'Internal server error' }),
      });

      await expect(getDatasets()).rejects.toThrow('Internal server error');
    });

    it('should handle network errors', async () => {
      (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

      await expect(getDatasets()).rejects.toThrow('Network error');
    });

    it('should handle malformed error responses', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => {
          throw new Error('Invalid JSON');
        },
      });

      await expect(getDatasets()).rejects.toThrow('HTTP error! status: 404');
    });
  });

  describe('getDataset', () => {
    it('should fetch a single dataset by ID', async () => {
      const mockDataset = {
        id: 'test-123',
        name: 'test-dataset',
        status: DatasetStatus.READY,
        progress: 100,
        created_at: new Date().toISOString(),
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockDataset,
      });

      const result = await getDataset('test-123');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/datasets/test-123'),
        expect.anything()
      );
      expect(result).toEqual(mockDataset);
    });

    it('should handle 404 errors', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Dataset not found' }),
      });

      await expect(getDataset('nonexistent')).rejects.toThrow(
        'Dataset not found'
      );
    });
  });

  describe('downloadDataset', () => {
    it('should download dataset with repo_id only', async () => {
      const mockDataset = {
        id: 'new-123',
        name: 'downloaded-dataset',
        status: DatasetStatus.DOWNLOADING,
        progress: 0,
        created_at: new Date().toISOString(),
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockDataset,
      });

      const result = await downloadDataset({ repo_id: 'test/dataset' });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/datasets/download'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('test/dataset'),
        })
      );
      expect(result).toEqual(mockDataset);
    });

    it('should download dataset with access token', async () => {
      const mockDataset = {
        id: 'private-123',
        name: 'private-dataset',
        status: DatasetStatus.DOWNLOADING,
        progress: 0,
        created_at: new Date().toISOString(),
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockDataset,
      });

      await downloadDataset({
        repo_id: 'test/private-dataset',
        access_token: 'hf_token',
      });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('hf_token'),
        })
      );
    });

    it('should download dataset with split and config', async () => {
      const mockDataset = {
        id: 'config-123',
        name: 'configured-dataset',
        status: DatasetStatus.DOWNLOADING,
        progress: 0,
        created_at: new Date().toISOString(),
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockDataset,
      });

      await downloadDataset({
        repo_id: 'test/dataset',
        split: 'train',
        config: 'default',
      });

      const callBody = (global.fetch as any).mock.calls[0][1].body;
      const parsedBody = JSON.parse(callBody);

      expect(parsedBody.split).toBe('train');
      expect(parsedBody.config).toBe('default');
    });

    it('should handle download errors', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'Invalid repository ID' }),
      });

      await expect(
        downloadDataset({ repo_id: 'invalid' })
      ).rejects.toThrow('Invalid repository ID');
    });
  });

  describe('deleteDataset', () => {
    it('should delete dataset by ID', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

      await deleteDataset('delete-123');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/datasets/delete-123'),
        expect.objectContaining({
          method: 'DELETE',
        })
      );
    });

    it('should handle delete errors', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: async () => ({ detail: 'Dataset has dependencies' }),
      });

      await expect(deleteDataset('protected-123')).rejects.toThrow(
        'Dataset has dependencies'
      );
    });
  });

  describe('getDatasetSamples', () => {
    it('should fetch samples without parameters', async () => {
      const mockResponse = {
        data: [{ id: '1', text: 'Sample text' }],
        total: 1,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await getDatasetSamples('dataset-123');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/datasets/dataset-123/samples'),
        expect.anything()
      );
      expect(result).toEqual(mockResponse);
    });

    it('should fetch samples with pagination', async () => {
      const mockResponse = {
        data: [],
        total: 0,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      await getDatasetSamples('dataset-123', { skip: 20, limit: 10 });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('skip=20'),
        expect.anything()
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('limit=10'),
        expect.anything()
      );
    });

    it('should fetch samples with search', async () => {
      const mockResponse = {
        data: [],
        total: 0,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      await getDatasetSamples('dataset-123', { search: 'keyword' });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('search=keyword'),
        expect.anything()
      );
    });

    it('should handle samples fetch errors', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Dataset not found' }),
      });

      await expect(getDatasetSamples('nonexistent')).rejects.toThrow(
        'Dataset not found'
      );
    });
  });

  describe('getDatasetStatistics', () => {
    it('should fetch dataset statistics', async () => {
      const mockStats = {
        num_samples: 1000,
        num_tokens: 250000,
        avg_seq_length: 250.5,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockStats,
      });

      const result = await getDatasetStatistics('dataset-123');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/datasets/dataset-123/statistics'),
        expect.anything()
      );
      expect(result).toEqual(mockStats);
    });

    it('should handle statistics fetch errors', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Statistics not available' }),
      });

      await expect(getDatasetStatistics('dataset-123')).rejects.toThrow(
        'Statistics not available'
      );
    });
  });

  describe('tokenizeDataset', () => {
    it('should tokenize dataset with settings', async () => {
      const mockDataset = {
        id: 'dataset-123',
        name: 'tokenized-dataset',
        status: DatasetStatus.PROCESSING,
        progress: 0,
        created_at: new Date().toISOString(),
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockDataset,
      });

      const result = await tokenizeDataset('dataset-123', {
        max_length: 512,
        truncation: true,
        padding: true,
        add_special_tokens: true,
      });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/datasets/dataset-123/tokenize'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('512'),
        })
      );
      expect(result).toEqual(mockDataset);
    });

    it('should tokenize dataset with minimal settings', async () => {
      const mockDataset = {
        id: 'dataset-123',
        name: 'tokenized-dataset',
        status: DatasetStatus.PROCESSING,
        progress: 0,
        created_at: new Date().toISOString(),
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockDataset,
      });

      await tokenizeDataset('dataset-123', {});

      expect(global.fetch).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({
          method: 'POST',
        })
      );
    });

    it('should handle tokenization errors', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: async () => ({ detail: 'Dataset is already being tokenized' }),
      });

      await expect(
        tokenizeDataset('dataset-123', { max_length: 512 })
      ).rejects.toThrow('Dataset is already being tokenized');
    });
  });

  describe('authentication', () => {
    it('should not include Authorization header when no token', async () => {
      mockLocalStorage.getItem.mockReturnValue(null);

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [], total: 0 }),
      });

      await getDatasets();

      const headers = (global.fetch as any).mock.calls[0][1].headers;
      expect(headers.Authorization).toBeUndefined();
    });

    it('should include Authorization header with token', async () => {
      mockLocalStorage.getItem.mockReturnValue('my_auth_token');

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [], total: 0 }),
      });

      await getDatasets();

      const headers = (global.fetch as any).mock.calls[0][1].headers;
      expect(headers.Authorization).toBe('Bearer my_auth_token');
    });
  });

  describe('error handling', () => {
    it('should extract detail from error response', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'Validation error' }),
      });

      await expect(getDatasets()).rejects.toThrow('Validation error');
    });

    it('should extract message from error response', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ message: 'Server error' }),
      });

      await expect(getDatasets()).rejects.toThrow('Server error');
    });

    it('should fallback to generic error for unknown responses', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 503,
        json: async () => ({}),
      });

      await expect(getDatasets()).rejects.toThrow('API request failed');
    });

    it('should handle non-JSON error responses', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => {
          throw new Error('Not JSON');
        },
      });

      await expect(getDatasets()).rejects.toThrow('HTTP error! status: 500');
    });
  });
});
