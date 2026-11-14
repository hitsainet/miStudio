/**
 * Zustand store for extraction template management.
 *
 * This store manages the global state for extraction templates, including:
 * - List of templates
 * - Loading states
 * - Error handling
 * - CRUD operations
 * - Favorite management
 * - Export/Import functionality
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import {
  ExtractionTemplate,
  ExtractionTemplateCreate,
  ExtractionTemplateUpdate,
  ExtractionTemplateListParams,
  ExtractionTemplateExport,
  ExtractionTemplateImportResult,
} from '../types/extractionTemplate';
import * as api from '../api/extractionTemplates';

interface ExtractionTemplatesState {
  // State
  templates: ExtractionTemplate[];
  favorites: ExtractionTemplate[];
  selectedTemplate: ExtractionTemplate | null;
  loading: boolean;
  error: string | null;
  pagination: {
    page: number;
    limit: number;
    total: number;
    total_pages: number;
    has_next: boolean;
    has_prev: boolean;
  } | null;

  // Actions
  fetchTemplates: (params?: ExtractionTemplateListParams) => Promise<void>;
  fetchTemplate: (id: string) => Promise<void>;
  fetchFavorites: () => Promise<void>;
  createTemplate: (template: ExtractionTemplateCreate) => Promise<ExtractionTemplate>;
  updateTemplate: (id: string, updates: ExtractionTemplateUpdate) => Promise<ExtractionTemplate>;
  deleteTemplate: (id: string) => Promise<void>;
  toggleFavorite: (id: string) => Promise<ExtractionTemplate>;
  exportTemplates: (templateIds?: string[]) => Promise<ExtractionTemplateExport>;
  importTemplates: (importData: ExtractionTemplateExport, overwriteDuplicates?: boolean) => Promise<ExtractionTemplateImportResult>;
  setSelectedTemplate: (template: ExtractionTemplate | null) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
}

export const useExtractionTemplatesStore = create<ExtractionTemplatesState>()(
  devtools(
    (set, _get) => ({
      // Initial state
      templates: [],
      favorites: [],
      selectedTemplate: null,
      loading: false,
      error: null,
      pagination: null,

      // Fetch all templates with optional filters
      fetchTemplates: async (params?: ExtractionTemplateListParams) => {
        set({ loading: true, error: null });
        try {
          const response = await api.getExtractionTemplates(params);
          set({
            templates: response.data || [],
            pagination: response.pagination || null,
            loading: false,
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch templates';
          set({ error: errorMessage, loading: false });
        }
      },

      // Fetch a single template by ID
      fetchTemplate: async (id: string) => {
        set({ loading: true, error: null });
        try {
          const template = await api.getExtractionTemplate(id);
          set({
            selectedTemplate: template,
            loading: false,
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch template';
          set({ error: errorMessage, loading: false });
        }
      },

      // Fetch favorite templates
      fetchFavorites: async () => {
        set({ loading: true, error: null });
        try {
          const response = await api.getFavoriteExtractionTemplates();
          set({
            favorites: response.data || [],
            loading: false,
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch favorites';
          set({ error: errorMessage, loading: false });
        }
      },

      // Create a new template
      createTemplate: async (template: ExtractionTemplateCreate) => {
        set({ loading: true, error: null });
        try {
          const newTemplate = await api.createExtractionTemplate(template);
          set((state) => ({
            templates: [...state.templates, newTemplate],
            loading: false,
          }));
          return newTemplate;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to create template';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Update an existing template
      updateTemplate: async (id: string, updates: ExtractionTemplateUpdate) => {
        set({ loading: true, error: null });
        try {
          const updatedTemplate = await api.updateExtractionTemplate(id, updates);
          set((state) => ({
            templates: state.templates.map((t) => (t.id === id ? updatedTemplate : t)),
            selectedTemplate: state.selectedTemplate?.id === id ? updatedTemplate : state.selectedTemplate,
            loading: false,
          }));
          return updatedTemplate;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to update template';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Delete a template
      deleteTemplate: async (id: string) => {
        set({ loading: true, error: null });
        try {
          await api.deleteExtractionTemplate(id);
          set((state) => ({
            templates: state.templates.filter((t) => t.id !== id),
            favorites: state.favorites.filter((t) => t.id !== id),
            selectedTemplate: state.selectedTemplate?.id === id ? null : state.selectedTemplate,
            loading: false,
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to delete template';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Toggle favorite status
      toggleFavorite: async (id: string) => {
        set({ loading: true, error: null });
        try {
          const updatedTemplate = await api.toggleExtractionTemplateFavorite(id);
          set((state) => ({
            templates: state.templates.map((t) => (t.id === id ? updatedTemplate : t)),
            favorites: updatedTemplate.is_favorite
              ? [...state.favorites, updatedTemplate]
              : state.favorites.filter((t) => t.id !== id),
            selectedTemplate: state.selectedTemplate?.id === id ? updatedTemplate : state.selectedTemplate,
            loading: false,
          }));
          return updatedTemplate;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to toggle favorite';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Export templates to JSON
      exportTemplates: async (templateIds?: string[]) => {
        set({ loading: true, error: null });
        try {
          const exportData = await api.exportExtractionTemplates(templateIds);
          set({ loading: false });
          return exportData;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to export templates';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Import templates from JSON
      importTemplates: async (importData: ExtractionTemplateExport, overwriteDuplicates: boolean = false) => {
        set({ loading: true, error: null });
        try {
          const result = await api.importExtractionTemplates(importData, overwriteDuplicates);

          // Refresh templates list after import
          const response = await api.getExtractionTemplates();
          set({
            templates: response.data || [],
            pagination: response.pagination || null,
            loading: false,
          });

          return result;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to import templates';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Set selected template
      setSelectedTemplate: (template: ExtractionTemplate | null) => {
        set({ selectedTemplate: template });
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
      name: 'ExtractionTemplatesStore',
    }
  )
);
