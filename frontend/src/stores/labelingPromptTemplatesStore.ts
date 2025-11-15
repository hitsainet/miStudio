/**
 * Zustand store for labeling prompt template management.
 *
 * This store manages the global state for labeling prompt templates, including:
 * - List of templates
 * - Loading states
 * - Error handling
 * - CRUD operations
 * - Default template management
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import {
  LabelingPromptTemplate,
  LabelingPromptTemplateCreate,
  LabelingPromptTemplateUpdate,
  LabelingPromptTemplateListParams,
} from '../types/labelingPromptTemplate';
import * as api from '../api/labelingPromptTemplates';

interface LabelingPromptTemplatesState {
  // State
  templates: LabelingPromptTemplate[];
  defaultTemplate: LabelingPromptTemplate | null;
  selectedTemplate: LabelingPromptTemplate | null;
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
  fetchTemplates: (params?: LabelingPromptTemplateListParams) => Promise<void>;
  fetchTemplate: (id: string) => Promise<void>;
  fetchDefaultTemplate: () => Promise<void>;
  createTemplate: (template: LabelingPromptTemplateCreate) => Promise<LabelingPromptTemplate>;
  updateTemplate: (id: string, updates: LabelingPromptTemplateUpdate) => Promise<LabelingPromptTemplate>;
  deleteTemplate: (id: string) => Promise<void>;
  setDefaultTemplate: (id: string) => Promise<void>;
  getUsageCount: (id: string) => Promise<number>;
  setSelectedTemplate: (template: LabelingPromptTemplate | null) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
}

export const useLabelingPromptTemplatesStore = create<LabelingPromptTemplatesState>()(
  devtools(
    (set, get) => ({
      // Initial state
      templates: [],
      defaultTemplate: null,
      selectedTemplate: null,
      loading: false,
      error: null,
      pagination: null,

      // Fetch all templates with optional filters
      fetchTemplates: async (params?: LabelingPromptTemplateListParams) => {
        set({ loading: true, error: null });
        try {
          const response = await api.getLabelingPromptTemplates(params);
          // Debug logging
          console.log('[LabelingPromptTemplatesStore] Fetched templates:',
            response.data?.map(t => ({
              id: t.id,
              name: t.name,
              user_prompt_first_100: t.user_prompt_template?.substring(0, 100)
            }))
          );
          set({
            templates: response.data || [],
            pagination: response.meta || null,
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
          const template = await api.getLabelingPromptTemplate(id);
          set({
            selectedTemplate: template,
            loading: false,
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch template';
          set({ error: errorMessage, loading: false });
        }
      },

      // Fetch the default template
      fetchDefaultTemplate: async () => {
        set({ loading: true, error: null });
        try {
          const template = await api.getDefaultLabelingPromptTemplate();
          set({
            defaultTemplate: template,
            loading: false,
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch default template';
          set({ error: errorMessage, loading: false });
        }
      },

      // Create a new template
      createTemplate: async (template: LabelingPromptTemplateCreate) => {
        set({ loading: true, error: null });
        try {
          const newTemplate = await api.createLabelingPromptTemplate(template);
          set((state) => ({
            templates: [...state.templates, newTemplate],
            loading: false,
          }));

          // If the new template is default, update defaultTemplate
          if (newTemplate.is_default) {
            set({ defaultTemplate: newTemplate });
          }

          return newTemplate;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to create template';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Update an existing template
      updateTemplate: async (id: string, updates: LabelingPromptTemplateUpdate) => {
        set({ loading: true, error: null });
        try {
          const updatedTemplate = await api.updateLabelingPromptTemplate(id, updates);
          set((state) => ({
            templates: state.templates.map((t) => (t.id === id ? updatedTemplate : t)),
            selectedTemplate: state.selectedTemplate?.id === id ? updatedTemplate : state.selectedTemplate,
            defaultTemplate: state.defaultTemplate?.id === id ? updatedTemplate : state.defaultTemplate,
            loading: false,
          }));

          // If the updated template is now default, update defaultTemplate
          if (updatedTemplate.is_default) {
            set({ defaultTemplate: updatedTemplate });
          }

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
          await api.deleteLabelingPromptTemplate(id);
          set((state) => ({
            templates: state.templates.filter((t) => t.id !== id),
            selectedTemplate: state.selectedTemplate?.id === id ? null : state.selectedTemplate,
            defaultTemplate: state.defaultTemplate?.id === id ? null : state.defaultTemplate,
            loading: false,
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to delete template';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Set a template as default
      setDefaultTemplate: async (id: string) => {
        set({ loading: true, error: null });
        try {
          await api.setDefaultLabelingPromptTemplate(id);

          // Update all templates to reflect new default
          set((state) => ({
            templates: state.templates.map((t) => ({
              ...t,
              is_default: t.id === id,
            })),
            defaultTemplate: state.templates.find((t) => t.id === id) || state.defaultTemplate,
            loading: false,
          }));

          // Refresh templates to ensure consistency
          await get().fetchTemplates();
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to set default template';
          set({ error: errorMessage, loading: false });
          throw error;
        }
      },

      // Get usage count for a template
      getUsageCount: async (id: string) => {
        try {
          const result = await api.getLabelingPromptTemplateUsageCount(id);
          return result.usage_count;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to get usage count';
          set({ error: errorMessage });
          throw error;
        }
      },

      // Set selected template
      setSelectedTemplate: (template: LabelingPromptTemplate | null) => {
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
    {
      name: 'labeling-prompt-templates-store',
    }
  )
);
