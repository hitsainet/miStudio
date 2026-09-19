/**
 * Tests for SAEs Store.
 *
 * Tests Zustand store functionality for SAE management including:
 * - State initialization
 * - Progress updates
 * - Filter management
 * - Pagination
 * - Error handling
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act } from '@testing-library/react';
import { useSAEsStore, selectReadySAEs, selectSAEsByModel } from './saesStore';
import { SAE, SAESource, SAEStatus, SAEFormat } from '../types/sae';

// Mock the API module
vi.mock('../api/saes', () => ({
  getSAEs: vi.fn(),
  getSAE: vi.fn(),
  deleteSAE: vi.fn(),
  deleteSAEsBatch: vi.fn(),
  previewHFRepository: vi.fn(),
  downloadSAE: vi.fn(),
  uploadSAE: vi.fn(),
  importSAEFromTraining: vi.fn(),
  importSAEFromFile: vi.fn(),
  browseSAEFeatures: vi.fn(),
}));

// Helper to create mock SAE
function createMockSAE(overrides: Partial<SAE> = {}): SAE {
  return {
    id: 'sae-1',
    name: 'Test SAE',
    description: 'A test SAE',
    source: SAESource.HUGGINGFACE,
    status: SAEStatus.READY,
    hf_repo_id: 'test/repo',
    hf_filepath: 'model.safetensors',
    hf_revision: 'main',
    training_id: null,
    model_name: 'gpt2',
    model_id: null,
    layer: 6,
    n_features: 16384,
    d_model: 768,
    architecture: 'standard',
    format: SAEFormat.SAELENS,
    local_path: '/data/saes/sae-1',
    file_size_bytes: 100000000,
    progress: 100,
    error_message: null,
    sae_metadata: {},
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    downloaded_at: '2024-01-01T00:00:00Z',
    ...overrides,
  };
}

describe('SAEsStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useSAEsStore.setState({
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
    });
  });

  describe('Initial state', () => {
    it('should have correct initial state', () => {
      const state = useSAEsStore.getState();

      expect(state.saes).toEqual([]);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
      expect(state.hfPreview).toBeNull();
      expect(state.filters.search).toBe('');
      expect(state.pagination.skip).toBe(0);
    });
  });

  describe('updateDownloadProgress', () => {
    it('should update progress for existing SAE', () => {
      const sae = createMockSAE({ id: 'sae-1', progress: 0, status: SAEStatus.DOWNLOADING });
      useSAEsStore.setState({ saes: [sae] });

      act(() => {
        useSAEsStore.getState().updateDownloadProgress('sae-1', 50, SAEStatus.DOWNLOADING, 'Downloading...');
      });

      const state = useSAEsStore.getState();
      expect(state.saes[0].progress).toBe(50);
      expect(state.saes[0].status).toBe(SAEStatus.DOWNLOADING);
    });

    it('should update status to ready when complete', () => {
      const sae = createMockSAE({ id: 'sae-1', progress: 90, status: SAEStatus.DOWNLOADING });
      useSAEsStore.setState({ saes: [sae] });

      act(() => {
        useSAEsStore.getState().updateDownloadProgress('sae-1', 100, SAEStatus.READY);
      });

      const state = useSAEsStore.getState();
      expect(state.saes[0].progress).toBe(100);
      expect(state.saes[0].status).toBe(SAEStatus.READY);
    });

    it('should not update non-existent SAE', () => {
      const sae = createMockSAE({ id: 'sae-1' });
      useSAEsStore.setState({ saes: [sae] });

      act(() => {
        useSAEsStore.getState().updateDownloadProgress('sae-nonexistent', 50);
      });

      const state = useSAEsStore.getState();
      expect(state.saes[0].progress).toBe(100); // Original value unchanged
    });
  });

  describe('updateSAEStatus', () => {
    it('should update SAE status', () => {
      const sae = createMockSAE({ id: 'sae-1', status: SAEStatus.DOWNLOADING });
      useSAEsStore.setState({ saes: [sae] });

      act(() => {
        useSAEsStore.getState().updateSAEStatus('sae-1', SAEStatus.READY);
      });

      const state = useSAEsStore.getState();
      expect(state.saes[0].status).toBe(SAEStatus.READY);
      expect(state.saes[0].progress).toBe(100);
    });

    it('should set error message on error status', () => {
      const sae = createMockSAE({ id: 'sae-1', status: SAEStatus.DOWNLOADING });
      useSAEsStore.setState({ saes: [sae] });

      act(() => {
        useSAEsStore.getState().updateSAEStatus('sae-1', SAEStatus.ERROR, 'Download failed');
      });

      const state = useSAEsStore.getState();
      expect(state.saes[0].status).toBe(SAEStatus.ERROR);
      expect(state.saes[0].error_message).toBe('Download failed');
    });
  });

  describe('setFilters', () => {
    it('should update filters and reset pagination', () => {
      useSAEsStore.setState({
        pagination: { skip: 20, limit: 20, total: 100, hasMore: true },
      });

      act(() => {
        useSAEsStore.getState().setFilters({ search: 'gpt2', source: SAESource.HUGGINGFACE });
      });

      const state = useSAEsStore.getState();
      expect(state.filters.search).toBe('gpt2');
      expect(state.filters.source).toBe(SAESource.HUGGINGFACE);
      expect(state.pagination.skip).toBe(0); // Reset to first page
    });

    it('should merge filters with existing', () => {
      useSAEsStore.setState({
        filters: {
          search: 'existing',
          source: SAESource.LOCAL,
          status: null,
          modelName: null,
          sortBy: 'created_at',
          order: 'desc',
        },
      });

      act(() => {
        useSAEsStore.getState().setFilters({ search: 'new' });
      });

      const state = useSAEsStore.getState();
      expect(state.filters.search).toBe('new');
      expect(state.filters.source).toBe(SAESource.LOCAL); // Unchanged
    });
  });

  describe('setPage', () => {
    it('should update pagination skip', () => {
      act(() => {
        useSAEsStore.getState().setPage(40);
      });

      const state = useSAEsStore.getState();
      expect(state.pagination.skip).toBe(40);
    });
  });

  describe('setError / clearError', () => {
    it('should set error message', () => {
      act(() => {
        useSAEsStore.getState().setError('Something went wrong');
      });

      expect(useSAEsStore.getState().error).toBe('Something went wrong');
    });

    it('should clear error message', () => {
      useSAEsStore.setState({ error: 'Some error' });

      act(() => {
        useSAEsStore.getState().clearError();
      });

      expect(useSAEsStore.getState().error).toBeNull();
    });
  });

  describe('clearHFPreview', () => {
    it('should clear HuggingFace preview', () => {
      useSAEsStore.setState({
        hfPreview: {
          repo_id: 'test/repo',
          repo_type: 'model',
          description: 'Test',
          files: [],
          sae_files: [],
          sae_paths: [],
          model_name: null,
          total_size_bytes: null,
        },
        hfPreviewError: 'Some error',
      });

      act(() => {
        useSAEsStore.getState().clearHFPreview();
      });

      const state = useSAEsStore.getState();
      expect(state.hfPreview).toBeNull();
      expect(state.hfPreviewError).toBeNull();
    });
  });

  describe('clearFeatureBrowser', () => {
    it('should clear feature browser state', () => {
      useSAEsStore.setState({
        featureBrowser: {
          saeId: 'sae-1',
          data: {
            sae_id: 'sae-1',
            n_features: 16384,
            features: [],
            pagination: { skip: 0, limit: 20, total: 16384, has_more: true },
          },
          loading: false,
          error: null,
        },
      });

      act(() => {
        useSAEsStore.getState().clearFeatureBrowser();
      });

      const state = useSAEsStore.getState();
      expect(state.featureBrowser.saeId).toBeNull();
      expect(state.featureBrowser.data).toBeNull();
    });
  });
});

describe('SAEsStore Selectors', () => {
  beforeEach(() => {
    useSAEsStore.setState({
      saes: [
        createMockSAE({ id: 'sae-1', status: SAEStatus.READY, model_name: 'gpt2' }),
        createMockSAE({ id: 'sae-2', status: SAEStatus.DOWNLOADING, model_name: 'gpt2' }),
        createMockSAE({ id: 'sae-3', status: SAEStatus.READY, model_name: 'llama-7b' }),
        createMockSAE({ id: 'sae-4', status: SAEStatus.ERROR, model_name: 'gpt2' }),
      ],
    });
  });

  describe('selectReadySAEs', () => {
    it('should return only ready SAEs', () => {
      const state = useSAEsStore.getState();
      const readySAEs = selectReadySAEs(state);

      expect(readySAEs).toHaveLength(2);
      expect(readySAEs.every(sae => sae.status === SAEStatus.READY)).toBe(true);
    });
  });

  describe('selectSAEsByModel', () => {
    it('should return ready SAEs for specific model', () => {
      const state = useSAEsStore.getState();
      const gpt2SAEs = selectSAEsByModel('gpt2')(state);

      expect(gpt2SAEs).toHaveLength(1);
      expect(gpt2SAEs[0].id).toBe('sae-1');
    });

    it('should return empty array for unknown model', () => {
      const state = useSAEsStore.getState();
      const unknownSAEs = selectSAEsByModel('unknown-model')(state);

      expect(unknownSAEs).toHaveLength(0);
    });
  });
});
