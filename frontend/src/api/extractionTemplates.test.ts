/**
 * Unit tests for extraction templates API client.
 *
 * These tests verify all API functions make correct HTTP requests
 * and handle responses/errors properly. Fetch calls are mocked.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  getExtractionTemplates,
  getExtractionTemplate,
  createExtractionTemplate,
  updateExtractionTemplate,
  deleteExtractionTemplate,
  toggleExtractionTemplateFavorite,
  getFavoriteExtractionTemplates,
  exportExtractionTemplates,
  importExtractionTemplates,
} from './extractionTemplates';
import { HookType } from '../types/extractionTemplate';

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

describe('extractionTemplates API client', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
    localStorageMock.getItem.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('getExtractionTemplates', () => {
    it('should fetch templates without parameters', async () => {
      const mockResponse = {
        data: [
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
        ],
        pagination: { page: 1, total: 1, total_pages: 1, has_next: false, has_prev: false },
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await getExtractionTemplates();

      expect(result).toEqual(mockResponse);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates',
        expect.objectContaining({
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
        })
      );
    });

    it('should fetch templates with query parameters', async () => {
      const mockResponse = {
        data: [],
        pagination: { page: 2, total: 0, total_pages: 0, has_next: false, has_prev: true },
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      await getExtractionTemplates({
        page: 2,
        limit: 10,
        search: 'layer',
        is_favorite: true,
      });

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates?page=2&limit=10&search=layer&is_favorite=true',
        expect.any(Object)
      );
    });

    it('should include auth token if present', async () => {
      localStorageMock.getItem.mockReturnValueOnce('test_auth_token');

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [], pagination: {} }),
      });

      await getExtractionTemplates();

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

  describe('getExtractionTemplate', () => {
    it('should fetch a single template by ID', async () => {
      const mockTemplate = {
        id: 'et_single',
        name: 'Single Template',
        description: 'Test description',
        layer_indices: [0, 5, 11, 23],
        hook_types: [HookType.RESIDUAL, HookType.MLP],
        max_samples: 1000,
        batch_size: 32,
        top_k_examples: 10,
        is_favorite: true,
        extra_metadata: { author: 'test' },
        created_at: '2025-10-13T00:00:00Z',
        updated_at: '2025-10-13T00:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockTemplate,
      });

      const result = await getExtractionTemplate('et_single');

      expect(result).toEqual(mockTemplate);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/et_single',
        expect.any(Object)
      );
    });

    it('should throw error if template not found', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Template not found' }),
      });

      await expect(getExtractionTemplate('et_nonexistent')).rejects.toThrow('Template not found');
    });
  });

  describe('createExtractionTemplate', () => {
    it('should create template with required fields', async () => {
      const mockTemplate = {
        id: 'et_created',
        name: 'New Template',
        description: 'Created template',
        layer_indices: [0, 5, 11],
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
        json: async () => mockTemplate,
      });

      const result = await createExtractionTemplate({
        name: 'New Template',
        description: 'Created template',
        layer_indices: [0, 5, 11],
        hook_types: [HookType.RESIDUAL],
        max_samples: 1000,
        batch_size: 32,
        top_k_examples: 10,
      });

      expect(result).toEqual(mockTemplate);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            name: 'New Template',
            description: 'Created template',
            layer_indices: [0, 5, 11],
            hook_types: ['residual'],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
          }),
        })
      );
    });

    it('should create template with favorite flag', async () => {
      const mockTemplate = {
        id: 'et_favorite',
        name: 'Favorite Template',
        layer_indices: [0],
        hook_types: [HookType.ATTENTION],
        max_samples: 500,
        batch_size: 16,
        top_k_examples: 5,
        is_favorite: true,
        created_at: '2025-10-13T00:00:00Z',
        updated_at: '2025-10-13T00:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockTemplate,
      });

      await createExtractionTemplate({
        name: 'Favorite Template',
        layer_indices: [0],
        hook_types: [HookType.ATTENTION],
        max_samples: 500,
        batch_size: 16,
        top_k_examples: 5,
        is_favorite: true,
      });

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates',
        expect.objectContaining({
          body: JSON.stringify({
            name: 'Favorite Template',
            layer_indices: [0],
            hook_types: ['attention'],
            max_samples: 500,
            batch_size: 16,
            top_k_examples: 5,
            is_favorite: true,
          }),
        })
      );
    });

    it('should create template with metadata', async () => {
      const mockTemplate = {
        id: 'et_metadata',
        name: 'Template with Metadata',
        layer_indices: [0, 5],
        hook_types: [HookType.MLP],
        max_samples: 1000,
        batch_size: 32,
        top_k_examples: 10,
        is_favorite: false,
        extra_metadata: { author: 'user', version: '1.0' },
        created_at: '2025-10-13T00:00:00Z',
        updated_at: '2025-10-13T00:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockTemplate,
      });

      await createExtractionTemplate({
        name: 'Template with Metadata',
        layer_indices: [0, 5],
        hook_types: [HookType.MLP],
        max_samples: 1000,
        batch_size: 32,
        top_k_examples: 10,
        extra_metadata: { author: 'user', version: '1.0' },
      });

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates',
        expect.objectContaining({
          body: JSON.stringify({
            name: 'Template with Metadata',
            layer_indices: [0, 5],
            hook_types: ['mlp'],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
            extra_metadata: { author: 'user', version: '1.0' },
          }),
        })
      );
    });

    it('should handle validation error', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'Invalid layer indices' }),
      });

      await expect(
        createExtractionTemplate({
          name: 'Invalid Template',
          layer_indices: [-1],
          hook_types: [HookType.RESIDUAL],
          max_samples: 1000,
          batch_size: 32,
          top_k_examples: 10,
        })
      ).rejects.toThrow('Invalid layer indices');
    });
  });

  describe('updateExtractionTemplate', () => {
    it('should update template name', async () => {
      const mockUpdatedTemplate = {
        id: 'et_update',
        name: 'Updated Name',
        layer_indices: [0, 5],
        hook_types: [HookType.RESIDUAL],
        max_samples: 1000,
        batch_size: 32,
        top_k_examples: 10,
        is_favorite: false,
        created_at: '2025-10-13T00:00:00Z',
        updated_at: '2025-10-13T01:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockUpdatedTemplate,
      });

      const result = await updateExtractionTemplate('et_update', {
        name: 'Updated Name',
      });

      expect(result).toEqual(mockUpdatedTemplate);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/et_update',
        expect.objectContaining({
          method: 'PATCH',
          body: JSON.stringify({ name: 'Updated Name' }),
        })
      );
    });

    it('should update multiple fields', async () => {
      const mockUpdatedTemplate = {
        id: 'et_multi',
        name: 'Multi Update',
        description: 'Updated description',
        layer_indices: [0, 11, 23],
        hook_types: [HookType.RESIDUAL, HookType.ATTENTION],
        max_samples: 2000,
        batch_size: 64,
        top_k_examples: 20,
        is_favorite: true,
        created_at: '2025-10-13T00:00:00Z',
        updated_at: '2025-10-13T01:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockUpdatedTemplate,
      });

      await updateExtractionTemplate('et_multi', {
        description: 'Updated description',
        layer_indices: [0, 11, 23],
        hook_types: [HookType.RESIDUAL, HookType.ATTENTION],
        max_samples: 2000,
        batch_size: 64,
        top_k_examples: 20,
        is_favorite: true,
      });

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/et_multi',
        expect.objectContaining({
          body: JSON.stringify({
            description: 'Updated description',
            layer_indices: [0, 11, 23],
            hook_types: ['residual', 'attention'],
            max_samples: 2000,
            batch_size: 64,
            top_k_examples: 20,
            is_favorite: true,
          }),
        })
      );
    });

    it('should throw error if template not found', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Template not found' }),
      });

      await expect(
        updateExtractionTemplate('et_nonexistent', { name: 'New Name' })
      ).rejects.toThrow('Template not found');
    });
  });

  describe('deleteExtractionTemplate', () => {
    it('should delete a template successfully', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        status: 204,
      });

      const result = await deleteExtractionTemplate('et_delete');

      expect(result).toBeUndefined();
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/et_delete',
        expect.objectContaining({
          method: 'DELETE',
        })
      );
    });

    it('should throw error if delete fails', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Template not found' }),
      });

      await expect(deleteExtractionTemplate('et_missing')).rejects.toThrow('Template not found');
    });
  });

  describe('toggleExtractionTemplateFavorite', () => {
    it('should toggle favorite status', async () => {
      const mockToggledTemplate = {
        id: 'et_toggle',
        name: 'Toggle Favorite',
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
        json: async () => mockToggledTemplate,
      });

      const result = await toggleExtractionTemplateFavorite('et_toggle');

      expect(result).toEqual(mockToggledTemplate);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/et_toggle/favorite',
        expect.objectContaining({
          method: 'POST',
        })
      );
    });

    it('should throw error if template not found', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Template not found' }),
      });

      await expect(toggleExtractionTemplateFavorite('et_missing')).rejects.toThrow(
        'Template not found'
      );
    });
  });

  describe('getFavoriteExtractionTemplates', () => {
    it('should fetch favorite templates', async () => {
      const mockResponse = {
        data: [
          {
            id: 'et_fav1',
            name: 'Favorite 1',
            layer_indices: [0, 5],
            hook_types: [HookType.RESIDUAL],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
            is_favorite: true,
            created_at: '2025-10-13T00:00:00Z',
            updated_at: '2025-10-13T00:00:00Z',
          },
          {
            id: 'et_fav2',
            name: 'Favorite 2',
            layer_indices: [11, 23],
            hook_types: [HookType.MLP],
            max_samples: 500,
            batch_size: 16,
            top_k_examples: 5,
            is_favorite: true,
            created_at: '2025-10-13T00:00:00Z',
            updated_at: '2025-10-13T00:00:00Z',
          },
        ],
        pagination: { page: 1, total: 2, total_pages: 1, has_next: false, has_prev: false },
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await getFavoriteExtractionTemplates();

      expect(result).toEqual(mockResponse);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/favorites',
        expect.any(Object)
      );
    });

    it('should return empty response if no favorites', async () => {
      const mockResponse = {
        data: [],
        pagination: { page: 1, total: 0, total_pages: 0, has_next: false, has_prev: false },
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await getFavoriteExtractionTemplates();

      expect(result).toEqual(mockResponse);
      expect(result.data).toEqual([]);
    });
  });

  describe('exportExtractionTemplates', () => {
    it('should export all templates', async () => {
      const mockExportData = {
        version: '1.0',
        exported_at: '2025-10-13T00:00:00Z',
        count: 2,
        templates: [
          {
            name: 'Template 1',
            description: 'First template',
            layer_indices: [0, 5],
            hook_types: ['residual'],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
            is_favorite: false,
          },
          {
            name: 'Template 2',
            description: 'Second template',
            layer_indices: [11, 23],
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

      const result = await exportExtractionTemplates();

      expect(result).toEqual(mockExportData);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/export',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(null),
        })
      );
    });

    it('should export specific templates by ID', async () => {
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

      const result = await exportExtractionTemplates(['et_specific']);

      expect(result).toEqual(mockExportData);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/export',
        expect.objectContaining({
          body: JSON.stringify(['et_specific']),
        })
      );
    });
  });

  describe('importExtractionTemplates', () => {
    it('should import templates successfully', async () => {
      const importData = {
        version: '1.0',
        templates: [
          {
            name: 'Imported Template 1',
            layer_indices: [0, 5],
            hook_types: ['residual'],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
          },
          {
            name: 'Imported Template 2',
            layer_indices: [11],
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

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockResult,
      });

      const result = await importExtractionTemplates(importData);

      expect(result).toEqual(mockResult);
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/extraction-templates/import',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            ...importData,
            overwrite_duplicates: false,
          }),
        })
      );
    });

    it('should import with overwrite flag', async () => {
      const importData = {
        version: '1.0',
        templates: [
          {
            name: 'Duplicate Template',
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

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockResult,
      });

      await importExtractionTemplates(importData, true);

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
            name: 'Good Template',
            layer_indices: [0],
            hook_types: ['residual'],
            max_samples: 1000,
            batch_size: 32,
            top_k_examples: 10,
          },
          {
            name: 'Bad Template',
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
        errors: ['Template "Bad Template": Invalid layer indices'],
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockResult,
      });

      const result = await importExtractionTemplates(importData);

      expect(result.created).toBe(1);
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0]).toContain('Bad Template');
    });

    it('should handle import validation error', async () => {
      const invalidData = {
        version: '1.0',
        templates: 'not an array',
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'Invalid import data format' }),
      });

      await expect(importExtractionTemplates(invalidData as any)).rejects.toThrow(
        'Invalid import data format'
      );
    });
  });

  describe('error handling', () => {
    it('should handle network error', async () => {
      global.fetch = vi.fn().mockRejectedValueOnce(new Error('Network error'));

      await expect(getExtractionTemplates()).rejects.toThrow('Network error');
    });

    it('should handle JSON parse error', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => {
          throw new Error('Invalid JSON');
        },
      });

      await expect(getExtractionTemplates()).rejects.toThrow('HTTP error! status: 500');
    });

    it('should handle 204 No Content response', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        status: 204,
      });

      const result = await deleteExtractionTemplate('et_delete');
      expect(result).toBeUndefined();
    });

    it('should handle 409 Conflict for duplicate name', async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: async () => ({ detail: 'Template with this name already exists' }),
      });

      await expect(
        createExtractionTemplate({
          name: 'Duplicate',
          layer_indices: [0],
          hook_types: [HookType.RESIDUAL],
          max_samples: 1000,
          batch_size: 32,
          top_k_examples: 10,
        })
      ).rejects.toThrow('Template with this name already exists');
    });
  });
});
