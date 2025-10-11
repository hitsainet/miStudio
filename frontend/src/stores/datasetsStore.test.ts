/**
 * Unit tests for datasetsStore (Zustand store).
 *
 * This module tests the global state management for datasets,
 * including CRUD operations, loading states, and error handling.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { useDatasetsStore, setDatasetSubscriptionCallback } from './datasetsStore';
import { DatasetStatus } from '../types/dataset';

// Mock fetch globally
global.fetch = vi.fn();

describe('datasetsStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useDatasetsStore.setState({
      datasets: [],
      loading: false,
      error: null,
    });

    // Reset fetch mock
    vi.clearAllMocks();
  });

  afterEach(() => {
    // Clean up any timers
    vi.clearAllTimers();
  });

  describe('Initial State', () => {
    it('should have empty datasets array initially', () => {
      const { datasets } = useDatasetsStore.getState();
      expect(datasets).toEqual([]);
    });

    it('should not be loading initially', () => {
      const { loading } = useDatasetsStore.getState();
      expect(loading).toBe(false);
    });

    it('should have no error initially', () => {
      const { error } = useDatasetsStore.getState();
      expect(error).toBeNull();
    });
  });

  describe('fetchDatasets', () => {
    it('should fetch datasets successfully', async () => {
      const mockDatasets = [
        {
          id: '1',
          name: 'test-dataset',
          status: DatasetStatus.READY,
          progress: 100,
          created_at: new Date().toISOString(),
        },
      ];

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: mockDatasets }),
      });

      const { fetchDatasets } = useDatasetsStore.getState();
      await fetchDatasets();

      const state = useDatasetsStore.getState();
      expect(state.datasets).toEqual(mockDatasets);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('should set loading to true during fetch', async () => {
      (global.fetch as any).mockImplementation(
        () => new Promise((resolve) => setTimeout(resolve, 100))
      );

      const { fetchDatasets } = useDatasetsStore.getState();
      const fetchPromise = fetchDatasets();

      // Check loading state during fetch
      const state = useDatasetsStore.getState();
      expect(state.loading).toBe(true);

      await fetchPromise;
    });

    it('should handle fetch error', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 500,
      });

      const { fetchDatasets } = useDatasetsStore.getState();
      await fetchDatasets();

      const state = useDatasetsStore.getState();
      expect(state.error).toBeTruthy();
      expect(state.loading).toBe(false);
    });

    it('should handle network error', async () => {
      (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

      const { fetchDatasets } = useDatasetsStore.getState();
      await fetchDatasets();

      const state = useDatasetsStore.getState();
      expect(state.error).toBe('Network error');
      expect(state.loading).toBe(false);
    });

    it('should handle empty data response', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [] }),
      });

      const { fetchDatasets } = useDatasetsStore.getState();
      await fetchDatasets();

      const state = useDatasetsStore.getState();
      expect(state.datasets).toEqual([]);
      expect(state.loading).toBe(false);
    });

    it('should handle missing data field in response', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

      const { fetchDatasets } = useDatasetsStore.getState();
      await fetchDatasets();

      const state = useDatasetsStore.getState();
      expect(state.datasets).toEqual([]);
      expect(state.loading).toBe(false);
    });
  });

  describe('downloadDataset', () => {
    beforeEach(() => {
      // Use fake timers for polling tests
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should download dataset successfully', async () => {
      const mockDataset = {
        id: '123',
        name: 'new-dataset',
        status: DatasetStatus.DOWNLOADING,
        progress: 0,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockDataset,
      });

      const { downloadDataset } = useDatasetsStore.getState();
      const downloadPromise = downloadDataset('test/repo');

      // Fast-forward past the download initiation
      await downloadPromise;

      const state = useDatasetsStore.getState();
      expect(state.datasets).toContainEqual(
        expect.objectContaining({ id: '123', name: 'new-dataset' })
      );
    });

    it('should include access_token in request when provided', async () => {
      const mockDataset = {
        id: '456',
        name: 'private-dataset',
        status: DatasetStatus.DOWNLOADING,
        progress: 0,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockDataset,
      });

      const { downloadDataset } = useDatasetsStore.getState();
      await downloadDataset('test/private-repo', 'test_token');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: expect.stringContaining('test_token'),
        })
      );
    });

    it('should include split and config in request when provided', async () => {
      const mockDataset = {
        id: '789',
        name: 'configured-dataset',
        status: DatasetStatus.DOWNLOADING,
        progress: 0,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockDataset,
      });

      const { downloadDataset } = useDatasetsStore.getState();
      await downloadDataset('test/dataset', undefined, 'train', 'default');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('train'),
        })
      );

      const callBody = JSON.parse(
        (global.fetch as any).mock.calls[0][1].body
      );
      expect(callBody.split).toBe('train');
      expect(callBody.config).toBe('default');
    });

    it('should handle download error', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        json: async () => ({ detail: 'Dataset not found' }),
      });

      const { downloadDataset } = useDatasetsStore.getState();

      await expect(downloadDataset('test/nonexistent')).rejects.toThrow(
        'Dataset not found'
      );

      const state = useDatasetsStore.getState();
      expect(state.error).toBe('Dataset not found');
      expect(state.loading).toBe(false);
    });

    it('should handle network error during download', async () => {
      (global.fetch as any).mockRejectedValueOnce(new Error('Network failed'));

      const { downloadDataset } = useDatasetsStore.getState();

      await expect(downloadDataset('test/repo')).rejects.toThrow(
        'Network failed'
      );

      const state = useDatasetsStore.getState();
      expect(state.error).toBe('Network failed');
    });

    it('should normalize status to lowercase', async () => {
      const mockDataset = {
        id: '999',
        name: 'uppercase-status',
        status: 'DOWNLOADING', // uppercase
        progress: 0,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockDataset,
      });

      const { downloadDataset } = useDatasetsStore.getState();
      await downloadDataset('test/repo');

      const state = useDatasetsStore.getState();
      const addedDataset = state.datasets.find((d) => d.id === '999');
      expect(addedDataset?.status).toBe('downloading');
    });

    it('should call subscription callback when provided', async () => {
      const mockCallback = vi.fn();
      setDatasetSubscriptionCallback(mockCallback);

      const mockDataset = {
        id: 'sub-test',
        name: 'subscription-test',
        status: DatasetStatus.DOWNLOADING,
        progress: 0,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockDataset,
      });

      const { downloadDataset } = useDatasetsStore.getState();
      await downloadDataset('test/repo');

      expect(mockCallback).toHaveBeenCalledWith('sub-test');

      // Clean up
      setDatasetSubscriptionCallback(null as any);
    });
  });

  describe('deleteDataset', () => {
    it('should delete dataset successfully', async () => {
      // Set up initial state with a dataset
      useDatasetsStore.setState({
        datasets: [
          {
            id: 'delete-me',
            name: 'to-be-deleted',
            status: DatasetStatus.READY,
            progress: 100,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            source: 'huggingface',
          },
        ],
      });

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
      });

      const { deleteDataset } = useDatasetsStore.getState();
      await deleteDataset('delete-me');

      const state = useDatasetsStore.getState();
      expect(state.datasets).toEqual([]);
      expect(state.loading).toBe(false);
    });

    it('should handle delete error', async () => {
      useDatasetsStore.setState({
        datasets: [
          {
            id: 'cannot-delete',
            name: 'protected',
            status: DatasetStatus.READY,
            progress: 100,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            source: 'huggingface',
          },
        ],
      });

      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
      });

      const { deleteDataset } = useDatasetsStore.getState();

      await expect(deleteDataset('cannot-delete')).rejects.toThrow(
        'Failed to delete dataset'
      );

      const state = useDatasetsStore.getState();
      expect(state.error).toBeTruthy();
      expect(state.datasets).toHaveLength(1); // Dataset should still be there
    });

    it('should use DELETE method with correct endpoint', async () => {
      useDatasetsStore.setState({
        datasets: [
          {
            id: 'test-id',
            name: 'test',
            status: DatasetStatus.READY,
            progress: 100,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            source: 'huggingface',
          },
        ],
      });

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
      });

      const { deleteDataset } = useDatasetsStore.getState();
      await deleteDataset('test-id');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/test-id'),
        expect.objectContaining({ method: 'DELETE' })
      );
    });
  });

  describe('updateDatasetProgress', () => {
    it('should update progress for existing dataset', () => {
      useDatasetsStore.setState({
        datasets: [
          {
            id: 'progress-test',
            name: 'test-dataset',
            status: DatasetStatus.DOWNLOADING,
            progress: 0,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            source: 'huggingface',
          },
        ],
      });

      const { updateDatasetProgress } = useDatasetsStore.getState();
      updateDatasetProgress('progress-test', 50);

      const state = useDatasetsStore.getState();
      const dataset = state.datasets.find((d) => d.id === 'progress-test');
      expect(dataset?.progress).toBe(50);
    });

    it('should not affect other datasets', () => {
      useDatasetsStore.setState({
        datasets: [
          {
            id: 'dataset-1',
            name: 'first',
            status: DatasetStatus.DOWNLOADING,
            progress: 0,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            source: 'huggingface',
          },
          {
            id: 'dataset-2',
            name: 'second',
            status: DatasetStatus.DOWNLOADING,
            progress: 0,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            source: 'huggingface',
          },
        ],
      });

      const { updateDatasetProgress } = useDatasetsStore.getState();
      updateDatasetProgress('dataset-1', 75);

      const state = useDatasetsStore.getState();
      const dataset1 = state.datasets.find((d) => d.id === 'dataset-1');
      const dataset2 = state.datasets.find((d) => d.id === 'dataset-2');

      expect(dataset1?.progress).toBe(75);
      expect(dataset2?.progress).toBe(0);
    });

    it('should handle updating non-existent dataset gracefully', () => {
      useDatasetsStore.setState({
        datasets: [
          {
            id: 'existing',
            name: 'test',
            status: DatasetStatus.READY,
            progress: 100,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            source: 'huggingface',
          },
        ],
      });

      const { updateDatasetProgress } = useDatasetsStore.getState();
      updateDatasetProgress('non-existent', 50);

      const state = useDatasetsStore.getState();
      expect(state.datasets).toHaveLength(1);
      expect(state.datasets[0].id).toBe('existing');
    });
  });

  describe('updateDatasetStatus', () => {
    it('should update status for existing dataset', () => {
      useDatasetsStore.setState({
        datasets: [
          {
            id: 'status-test',
            name: 'test-dataset',
            status: DatasetStatus.DOWNLOADING,
            progress: 50,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            source: 'huggingface',
          },
        ],
      });

      const { updateDatasetStatus } = useDatasetsStore.getState();
      updateDatasetStatus('status-test', DatasetStatus.PROCESSING);

      const state = useDatasetsStore.getState();
      const dataset = state.datasets.find((d) => d.id === 'status-test');
      expect(dataset?.status).toBe(DatasetStatus.PROCESSING);
    });

    it('should set progress to 100 when status is READY', () => {
      useDatasetsStore.setState({
        datasets: [
          {
            id: 'complete-test',
            name: 'test-dataset',
            status: DatasetStatus.PROCESSING,
            progress: 75,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            source: 'huggingface',
          },
        ],
      });

      const { updateDatasetStatus } = useDatasetsStore.getState();
      updateDatasetStatus('complete-test', DatasetStatus.READY);

      const state = useDatasetsStore.getState();
      const dataset = state.datasets.find((d) => d.id === 'complete-test');
      expect(dataset?.status).toBe(DatasetStatus.READY);
      expect(dataset?.progress).toBe(100);
    });

    it('should preserve progress for non-READY statuses', () => {
      useDatasetsStore.setState({
        datasets: [
          {
            id: 'progress-preserve',
            name: 'test-dataset',
            status: DatasetStatus.DOWNLOADING,
            progress: 30,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            source: 'huggingface',
          },
        ],
      });

      const { updateDatasetStatus } = useDatasetsStore.getState();
      updateDatasetStatus('progress-preserve', DatasetStatus.PROCESSING);

      const state = useDatasetsStore.getState();
      const dataset = state.datasets.find((d) => d.id === 'progress-preserve');
      expect(dataset?.progress).toBe(30);
    });

    it('should update error_message when provided', () => {
      useDatasetsStore.setState({
        datasets: [
          {
            id: 'error-test',
            name: 'test-dataset',
            status: DatasetStatus.DOWNLOADING,
            progress: 25,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            source: 'huggingface',
          },
        ],
      });

      const { updateDatasetStatus } = useDatasetsStore.getState();
      updateDatasetStatus(
        'error-test',
        DatasetStatus.ERROR,
        'Download failed'
      );

      const state = useDatasetsStore.getState();
      const dataset = state.datasets.find((d) => d.id === 'error-test');
      expect(dataset?.status).toBe(DatasetStatus.ERROR);
      expect(dataset?.error_message).toBe('Download failed');
    });
  });

  describe('setError', () => {
    it('should set error message', () => {
      const { setError } = useDatasetsStore.getState();
      setError('Test error');

      const state = useDatasetsStore.getState();
      expect(state.error).toBe('Test error');
    });

    it('should set error to null', () => {
      useDatasetsStore.setState({ error: 'Existing error' });

      const { setError } = useDatasetsStore.getState();
      setError(null);

      const state = useDatasetsStore.getState();
      expect(state.error).toBeNull();
    });
  });

  describe('clearError', () => {
    it('should clear error message', () => {
      useDatasetsStore.setState({ error: 'Some error' });

      const { clearError } = useDatasetsStore.getState();
      clearError();

      const state = useDatasetsStore.getState();
      expect(state.error).toBeNull();
    });

    it('should handle clearing when error is already null', () => {
      const { clearError } = useDatasetsStore.getState();
      clearError();

      const state = useDatasetsStore.getState();
      expect(state.error).toBeNull();
    });
  });

  describe('setDatasetSubscriptionCallback', () => {
    it('should set subscription callback', async () => {
      const callback = vi.fn();
      setDatasetSubscriptionCallback(callback);

      const mockDataset = {
        id: 'callback-test',
        name: 'test',
        status: DatasetStatus.DOWNLOADING,
        progress: 0,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockDataset,
      });

      const { downloadDataset } = useDatasetsStore.getState();
      await downloadDataset('test/repo');

      // Callback should have been called
      expect(callback).toHaveBeenCalledWith('callback-test');

      // Clean up
      setDatasetSubscriptionCallback(null as any);
    });
  });
});
