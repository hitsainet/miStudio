/**
 * Unit tests for extractionTemplatesStore.
 *
 * These tests verify the Zustand store's state management and API integration.
 * Fetch calls are mocked to test store behavior without requiring a live backend.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useExtractionTemplatesStore } from './extractionTemplatesStore';
import { HookType, ExtractionTemplate } from '../types/extractionTemplate';

// Mock the API_BASE_URL
vi.mock('../config/api', () => ({
  API_BASE_URL: 'http://localhost:8000',
}));

describe('extractionTemplatesStore', () => {
  // Reset store state before each test
  beforeEach(() => {
    useExtractionTemplatesStore.setState({
      templates: [],
      favorites: [],
      selectedTemplate: null,
      loading: false,
      error: null,
      pagination: null,
    });
    // Clear all fetch mocks
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('fetchTemplates', () => {
    it('should fetch templates successfully', async () => {
      const mockTemplates: ExtractionTemplate[] = [
        {
          id: 'et_test1',
          name: 'Test Template',
          description: 'A test template',
          layer_indices: [0, 5, 11],
          hook_types: [HookType.RESIDUAL],
          max_samples: 1000,
          batch_size: 32,
          top_k_examples: 10,
          is_favorite: false,
          created_at: '2025-10-13T00:00:00Z',
          updated_at: '2025-10-13T00:00:00Z',
        },
      ];

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: mockTemplates,
          pagination: { page: 1, total: 1, total_pages: 1, has_next: false, has_prev: false },
        }),
      });

      const { fetchTemplates } = useExtractionTemplatesStore.getState();
      await fetchTemplates();

      const state = useExtractionTemplatesStore.getState();
      expect(state.templates).toEqual(mockTemplates);
      expect(state.pagination).toEqual({
        page: 1,
        total: 1,
        total_pages: 1,
        has_next: false,
        has_prev: false,
      });
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates',
        expect.any(Object)
      );
    });

    it('should fetch templates with filters', async () => {
      const mockTemplates: ExtractionTemplate[] = [];

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: mockTemplates,
          pagination: { page: 2, total: 0, total_pages: 0, has_next: false, has_prev: true },
        }),
      });

      const { fetchTemplates } = useExtractionTemplatesStore.getState();
      await fetchTemplates({ page: 2, limit: 10, search: 'test', is_favorite: true });

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates?page=2&limit=10&search=test&is_favorite=true',
        expect.any(Object)
      );
    });

    it('should handle fetch error', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ detail: 'Internal server error' }),
      });

      const { fetchTemplates } = useExtractionTemplatesStore.getState();
      await fetchTemplates();

      const state = useExtractionTemplatesStore.getState();
      expect(state.templates).toEqual([]);
      expect(state.loading).toBe(false);
      expect(state.error).toBe('Internal server error');
    });

    it('should handle network error', async () => {
      global.fetch = vi.fn().mockRejectedValueOnce(new Error('Network error'));

      const { fetchTemplates } = useExtractionTemplatesStore.getState();
      await fetchTemplates();

      const state = useExtractionTemplatesStore.getState();
      expect(state.templates).toEqual([]);
      expect(state.loading).toBe(false);
      expect(state.error).toBe('Network error');
    });
  });

  describe('fetchFavorites', () => {
    it('should fetch favorite templates successfully', async () => {
      const mockFavorites: ExtractionTemplate[] = [
        {
          id: 'et_fav1',
          name: 'Favorite 1',
          layer_indices: [0, 5],
          hook_types: [HookType.RESIDUAL, HookType.MLP],
          max_samples: 1000,
          batch_size: 32,
          top_k_examples: 10,
          is_favorite: true,
          created_at: '2025-10-13T00:00:00Z',
          updated_at: '2025-10-13T00:00:00Z',
        },
      ];

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: mockFavorites,
          pagination: { page: 1, total: 1, total_pages: 1, has_next: false, has_prev: false },
        }),
      });

      const { fetchFavorites } = useExtractionTemplatesStore.getState();
      await fetchFavorites();

      const state = useExtractionTemplatesStore.getState();
      expect(state.favorites).toEqual(mockFavorites);
      expect(state.loading).toBe(false);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/favorites',
        expect.any(Object)
      );
    });

    it('should handle empty favorites', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          data: [],
          pagination: { page: 1, total: 0, total_pages: 0, has_next: false, has_prev: false },
        }),
      });

      const { fetchFavorites } = useExtractionTemplatesStore.getState();
      await fetchFavorites();

      const state = useExtractionTemplatesStore.getState();
      expect(state.favorites).toEqual([]);
    });
  });

  describe('createTemplate', () => {
    it('should create template successfully', async () => {
      const newTemplate: ExtractionTemplate = {
        id: 'et_new',
        name: 'New Template',
        description: 'Created template',
        layer_indices: [0, 11, 23],
        hook_types: [HookType.RESIDUAL],
        max_samples: 1000,
        batch_size: 32,
        top_k_examples: 10,
        is_favorite: false,
        created_at: '2025-10-13T00:00:00Z',
        updated_at: '2025-10-13T00:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => newTemplate,
      });

      const { createTemplate } = useExtractionTemplatesStore.getState();
      await createTemplate({
        name: 'New Template',
        description: 'Created template',
        layer_indices: [0, 11, 23],
        hook_types: [HookType.RESIDUAL],
        max_samples: 1000,
        batch_size: 32,
        top_k_examples: 10,
      });

      const state = useExtractionTemplatesStore.getState();
      expect(state.templates).toHaveLength(1);
      expect(state.templates[0].id).toBe('et_new');
      expect(state.loading).toBe(false);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            name: 'New Template',
            description: 'Created template',
            layer_indices: [0, 11, 23],
            hook_types: ['residual'],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
          }),
        })
      );
    });

    it('should handle create error', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'Invalid layer indices' }),
      });

      const { createTemplate } = useExtractionTemplatesStore.getState();
      await expect(
        createTemplate({
          name: 'Invalid',
          layer_indices: [-1],
          hook_types: [HookType.RESIDUAL],
          max_samples: 1000,
          batch_size: 32,
          top_k_examples: 10,
        })
      ).rejects.toThrow('Invalid layer indices');

      const state = useExtractionTemplatesStore.getState();
      expect(state.error).toBe('Invalid layer indices');
    });

    it('should handle duplicate name error', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: async () => ({ detail: 'Template with this name already exists' }),
      });

      const { createTemplate } = useExtractionTemplatesStore.getState();
      await expect(
        createTemplate({
          name: 'Duplicate',
          layer_indices: [0],
          hook_types: [HookType.RESIDUAL],
          max_samples: 1000,
          batch_size: 32,
          top_k_examples: 10,
        })
      ).rejects.toThrow('Template with this name already exists');

      const state = useExtractionTemplatesStore.getState();
      expect(state.error).toBe('Template with this name already exists');
    });
  });

  describe('updateTemplate', () => {
    it('should update template successfully', async () => {
      useExtractionTemplatesStore.setState({
        templates: [
          {
            id: 'et_update',
            name: 'Old Name',
            layer_indices: [0],
            hook_types: [HookType.RESIDUAL],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
            is_favorite: false,
            created_at: '2025-10-13T00:00:00Z',
            updated_at: '2025-10-13T00:00:00Z',
          },
        ],
      });

      const updatedTemplate: ExtractionTemplate = {
        id: 'et_update',
        name: 'Updated Name',
        layer_indices: [0, 5],
        hook_types: [HookType.RESIDUAL, HookType.MLP],
        max_samples: 2000,
        batch_size: 64,
        top_k_examples: 20,
        is_favorite: true,
        created_at: '2025-10-13T00:00:00Z',
        updated_at: '2025-10-13T01:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => updatedTemplate,
      });

      const { updateTemplate } = useExtractionTemplatesStore.getState();
      await updateTemplate('et_update', {
        name: 'Updated Name',
        layer_indices: [0, 5],
        hook_types: [HookType.RESIDUAL, HookType.MLP],
        max_samples: 2000,
        batch_size: 64,
        top_k_examples: 20,
        is_favorite: true,
      });

      const state = useExtractionTemplatesStore.getState();
      expect(state.templates[0].name).toBe('Updated Name');
      expect(state.templates[0].layer_indices).toEqual([0, 5]);
      expect(state.templates[0].max_samples).toBe(2000);
      expect(state.loading).toBe(false);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/et_update',
        expect.objectContaining({
          method: 'PATCH',
        })
      );
    });

    it('should handle update error', async () => {
      useExtractionTemplatesStore.setState({
        templates: [
          {
            id: 'et_exists',
            name: 'Existing',
            layer_indices: [0],
            hook_types: [HookType.RESIDUAL],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
            is_favorite: false,
            created_at: '2025-10-13T00:00:00Z',
            updated_at: '2025-10-13T00:00:00Z',
          },
        ],
      });

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Template not found' }),
      });

      const { updateTemplate } = useExtractionTemplatesStore.getState();
      await expect(
        updateTemplate('et_nonexistent', { name: 'New Name' })
      ).rejects.toThrow('Template not found');

      const state = useExtractionTemplatesStore.getState();
      expect(state.templates[0].name).toBe('Existing'); // Unchanged
    });
  });

  describe('deleteTemplate', () => {
    it('should delete template successfully', async () => {
      useExtractionTemplatesStore.setState({
        templates: [
          {
            id: 'et_delete',
            name: 'To Delete',
            layer_indices: [0],
            hook_types: [HookType.RESIDUAL],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
            is_favorite: false,
            created_at: '2025-10-13T00:00:00Z',
            updated_at: '2025-10-13T00:00:00Z',
          },
        ],
      });

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        status: 204,
      });

      const { deleteTemplate } = useExtractionTemplatesStore.getState();
      await deleteTemplate('et_delete');

      const state = useExtractionTemplatesStore.getState();
      expect(state.templates).toHaveLength(0);
      expect(state.loading).toBe(false);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/et_delete',
        expect.objectContaining({ method: 'DELETE' })
      );
    });

    it('should handle delete error', async () => {
      useExtractionTemplatesStore.setState({
        templates: [
          {
            id: 'et_protected',
            name: 'Protected',
            layer_indices: [0],
            hook_types: [HookType.RESIDUAL],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
            is_favorite: false,
            created_at: '2025-10-13T00:00:00Z',
            updated_at: '2025-10-13T00:00:00Z',
          },
        ],
      });

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Template not found' }),
      });

      const { deleteTemplate } = useExtractionTemplatesStore.getState();
      await expect(deleteTemplate('et_nonexistent')).rejects.toThrow('Template not found');

      const state = useExtractionTemplatesStore.getState();
      expect(state.templates).toHaveLength(1); // Still present
    });
  });

  describe('toggleFavorite', () => {
    it('should toggle favorite status', async () => {
      useExtractionTemplatesStore.setState({
        templates: [
          {
            id: 'et_toggle',
            name: 'Toggle Me',
            layer_indices: [0],
            hook_types: [HookType.RESIDUAL],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
            is_favorite: false,
            created_at: '2025-10-13T00:00:00Z',
            updated_at: '2025-10-13T00:00:00Z',
          },
        ],
      });

      const toggledTemplate: ExtractionTemplate = {
        id: 'et_toggle',
        name: 'Toggle Me',
        layer_indices: [0],
        hook_types: [HookType.RESIDUAL],
        max_samples: 1000,
        batch_size: 32,
        top_k_examples: 10,
        is_favorite: true,
        created_at: '2025-10-13T00:00:00Z',
        updated_at: '2025-10-13T01:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => toggledTemplate,
      });

      const { toggleFavorite } = useExtractionTemplatesStore.getState();
      await toggleFavorite('et_toggle');

      const state = useExtractionTemplatesStore.getState();
      expect(state.templates[0].is_favorite).toBe(true);
      expect(state.loading).toBe(false);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/et_toggle/favorite',
        expect.objectContaining({ method: 'POST' })
      );
    });

    it('should handle toggle error', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Template not found' }),
      });

      const { toggleFavorite } = useExtractionTemplatesStore.getState();
      await expect(toggleFavorite('et_missing')).rejects.toThrow('Template not found');

      const state = useExtractionTemplatesStore.getState();
      expect(state.error).toBe('Template not found');
    });
  });

  describe('exportTemplates', () => {
    it('should export all templates', async () => {
      const mockExportData = {
        version: '1.0',
        exported_at: '2025-10-13T00:00:00Z',
        count: 2,
        templates: [
          {
            name: 'Template 1',
            layer_indices: [0, 5],
            hook_types: ['residual'],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
            is_favorite: false,
          },
          {
            name: 'Template 2',
            layer_indices: [11],
            hook_types: ['mlp'],
            max_samples: 500,
            batch_size: 16,
            top_k_examples: 5,
            is_favorite: true,
          },
        ],
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockExportData,
      });

      const { exportTemplates } = useExtractionTemplatesStore.getState();
      const result = await exportTemplates();

      expect(result).toEqual(mockExportData);
      const state = useExtractionTemplatesStore.getState();
      expect(state.loading).toBe(false);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/export',
        expect.objectContaining({ method: 'POST' })
      );
    });

    it('should export specific templates', async () => {
      const mockExportData = {
        version: '1.0',
        exported_at: '2025-10-13T00:00:00Z',
        count: 1,
        templates: [
          {
            name: 'Specific Template',
            layer_indices: [0],
            hook_types: ['attention'],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
            is_favorite: false,
          },
        ],
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockExportData,
      });

      const { exportTemplates } = useExtractionTemplatesStore.getState();
      await exportTemplates(['et_specific']);

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/export',
        expect.objectContaining({
          body: JSON.stringify(['et_specific']),
        })
      );
    });
  });

  describe('importTemplates', () => {
    it('should import templates successfully', async () => {
      const importData = {
        version: '1.0',
        templates: [
          {
            name: 'Imported 1',
            layer_indices: [0],
            hook_types: ['residual'],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
          },
          {
            name: 'Imported 2',
            layer_indices: [5],
            hook_types: ['mlp'],
            max_samples: 500,
            batch_size: 16,
            top_k_examples: 5,
          },
        ],
      };

      const mockResult = {
        created: 2,
        skipped: 0,
        errors: [],
      };

      // Mock both the import call and the subsequent getExtractionTemplates call
      global.fetch = vi
        .fn()
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockResult,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: [],
            pagination: { page: 1, total: 0, total_pages: 0, has_next: false, has_prev: false },
          }),
        });

      const { importTemplates } = useExtractionTemplatesStore.getState();
      const result = await importTemplates(importData, false);

      expect(result).toEqual(mockResult);
      const state = useExtractionTemplatesStore.getState();
      expect(state.loading).toBe(false);
    });

    it('should import with overwrite flag', async () => {
      const importData = {
        version: '1.0',
        templates: [
          {
            name: 'Overwrite Me',
            layer_indices: [0],
            hook_types: ['residual'],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
          },
        ],
      };

      const mockResult = {
        created: 0,
        skipped: 0,
        errors: [],
      };

      // Mock both the import call and the subsequent getExtractionTemplates call
      global.fetch = vi
        .fn()
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockResult,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: [],
            pagination: { page: 1, total: 0, total_pages: 0, has_next: false, has_prev: false },
          }),
        });

      const { importTemplates } = useExtractionTemplatesStore.getState();
      await importTemplates(importData, true);

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/import',
        expect.objectContaining({
          body: JSON.stringify({
            ...importData,
            overwrite_duplicates: true,
          }),
        })
      );
    });

    it('should handle import with errors', async () => {
      const importData = {
        version: '1.0',
        templates: [
          {
            name: 'Good',
            layer_indices: [0],
            hook_types: ['residual'],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
          },
          {
            name: 'Bad',
            layer_indices: [-1],
            hook_types: ['invalid'],
            max_samples: 0,
            batch_size: 0,
            top_k_examples: 0,
          },
        ],
      };

      const mockResult = {
        created: 1,
        skipped: 0,
        errors: ['Template "Bad": Invalid configuration'],
      };

      // Mock both the import call and the subsequent getExtractionTemplates call
      global.fetch = vi
        .fn()
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockResult,
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: [],
            pagination: { page: 1, total: 0, total_pages: 0, has_next: false, has_prev: false },
          }),
        });

      const { importTemplates } = useExtractionTemplatesStore.getState();
      const result = await importTemplates(importData, false);

      expect(result.created).toBe(1);
      expect(result.errors).toHaveLength(1);
    });
  });

  describe('state management', () => {
    it('should select a template', () => {
      const template: ExtractionTemplate = {
        id: 'et_select',
        name: 'Selected',
        layer_indices: [0],
        hook_types: [HookType.RESIDUAL],
        max_samples: 1000,
        batch_size: 32,
        top_k_examples: 10,
        is_favorite: false,
        created_at: '2025-10-13T00:00:00Z',
        updated_at: '2025-10-13T00:00:00Z',
      };

      const { setSelectedTemplate } = useExtractionTemplatesStore.getState();
      setSelectedTemplate(template);

      const state = useExtractionTemplatesStore.getState();
      expect(state.selectedTemplate).toEqual(template);
    });

    it('should clear selected template', () => {
      useExtractionTemplatesStore.setState({
        selectedTemplate: {
          id: 'et_clear',
          name: 'To Clear',
          layer_indices: [0],
          hook_types: [HookType.RESIDUAL],
          max_samples: 1000,
          batch_size: 32,
          top_k_examples: 10,
          is_favorite: false,
          created_at: '2025-10-13T00:00:00Z',
          updated_at: '2025-10-13T00:00:00Z',
        },
      });

      const { setSelectedTemplate } = useExtractionTemplatesStore.getState();
      setSelectedTemplate(null);

      const state = useExtractionTemplatesStore.getState();
      expect(state.selectedTemplate).toBeNull();
    });

    it('should set error message', () => {
      const { setError } = useExtractionTemplatesStore.getState();
      setError('Test error');

      const state = useExtractionTemplatesStore.getState();
      expect(state.error).toBe('Test error');
    });

    it('should clear error message', () => {
      useExtractionTemplatesStore.setState({ error: 'Some error' });

      const { clearError } = useExtractionTemplatesStore.getState();
      clearError();

      const state = useExtractionTemplatesStore.getState();
      expect(state.error).toBeNull();
    });
  });
});
