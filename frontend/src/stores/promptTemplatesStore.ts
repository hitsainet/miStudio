/**
 * Zustand store for prompt template management.
 *
 * This store manages the global state for prompt templates, including:
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
  PromptTemplate,
  PromptTemplateCreate,
  PromptTemplateUpdate,
  PromptTemplateListParams,
  PromptTemplateExport,
  PromptTemplateImportResult,
} from '../types/promptTemplate';
import * as api from '../api/promptTemplates';

interface PromptTemplatesState {
  // State
  templates: PromptTemplate[];
  favorites: PromptTemplate[];
  selectedTemplate: PromptTemplate | null;
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
  fetchTemplates: (params?: PromptTemplateListParams) => Promise<void>;
  fetchTemplate: (id: string) => Promise<void>;
  fetchFavorites: () => Promise<void>;
  createTemplate: (template: PromptTemplateCreate) => Promise<PromptTemplate>;
  updateTemplate: (id: string, updates: PromptTemplateUpdate) => Promise<PromptTemplate>;
  deleteTemplate: (id: string) => Promise<void>;
  toggleFavorite: (id: string) => Promise<PromptTemplate>;
  duplicateTemplate: (id: string) => Promise<PromptTemplate>;
  exportTemplates: (templateIds?: string[]) => Promise<PromptTemplateExport>;
  importTemplates: (importData: PromptTemplateExport, overwriteDuplicates?: boolean) => Promise<PromptTemplateImportResult>;
  setSelectedTemplate: (template: PromptTemplate | null) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
}

export const usePromptTemplatesStore = create<PromptTemplatesState>()(
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
      fetchTemplates: async (params?: PromptTemplateListParams) => {
        set({ loading: true, error: null });
        try {
          const response = await api.getPromptTemplates(params);
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
          const template = await api.getPromptTemplate(id);
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
          const response = await api.getFavoritePromptTemplates();
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
      createTemplate: async (template: PromptTemplateCreate) => {
        set({ loading: true, error: null });
        try {
          const newTemplate = await api.createPromptTemplate(template);
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
      updateTemplate: async (id: string, updates: PromptTemplateUpdate) => {
        set({ loading: true, error: null });
        try {
          const updatedTemplate = await api.updatePromptTemplate(id, updates);
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
          await api.deletePromptTemplate(id);
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
          const updatedTemplate = await api.togglePromptTemplateFavorite(id);
          set((state) => ({
            templates: state.templates.map((t) => (t.id === id ? updatedTemplate : t)),
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

      // Duplicate a template
      duplicateTemplate: async (id: string) => {
        set({ loading: true, error: null });
        try {
          const newTemplate = await api.duplicatePromptTemplate(id);
          set((state) => ({
            templates: [...state.templates, newTemplate],
            loading: false,
          }));
          return newTemplate;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to duplicate template';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Export templates
      exportTemplates: async (templateIds?: string[]) => {
        set({ loading: true, error: null });
        try {
          const exportData = await api.exportPromptTemplates(templateIds);
          set({ loading: false });
          return exportData;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to export templates';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Import templates
      importTemplates: async (
        importData: PromptTemplateExport,
        overwriteDuplicates: boolean = false
      ) => {
        set({ loading: true, error: null });
        try {
          const result = await api.importPromptTemplates(importData, overwriteDuplicates);
          // Refresh the templates list after import
          const response = await api.getPromptTemplates();
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
      setSelectedTemplate: (template: PromptTemplate | null) => {
        set({ selectedTemplate: template });
      },

      // Set error
      setError: (error: string | null) => {
        set({ error });
      },

      // Clear error
      clearError: () => {
        set({ error: null });
      },
    }),
    { name: 'PromptTemplatesStore' }
  )
);
