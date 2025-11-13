/**
 * API client for labeling prompt template operations.
 *
 * This module provides functions to interact with the backend labeling prompt templates API.
 */

import {
  LabelingPromptTemplate,
  LabelingPromptTemplateCreate,
  LabelingPromptTemplateUpdate,
  LabelingPromptTemplateListResponse,
  LabelingPromptTemplateListParams,
  LabelingPromptTemplateDeleteResponse,
  LabelingPromptTemplateSetDefaultResponse,
} from '../types/labelingPromptTemplate';
import { fetchAPI, buildQueryString } from './client';

/**
 * Get list of labeling prompt templates with optional filters
 */
export async function getLabelingPromptTemplates(
  params?: LabelingPromptTemplateListParams
): Promise<LabelingPromptTemplateListResponse> {
  const query = params ? buildQueryString(params) : '';
  const endpoint = `/labeling-prompt-templates${query ? `?${query}` : ''}`;
  return fetchAPI<LabelingPromptTemplateListResponse>(endpoint);
}

/**
 * Get a single labeling prompt template by ID
 */
export async function getLabelingPromptTemplate(
  id: string
): Promise<LabelingPromptTemplate> {
  return fetchAPI<LabelingPromptTemplate>(`/labeling-prompt-templates/${id}`);
}

/**
 * Get the default labeling prompt template
 */
export async function getDefaultLabelingPromptTemplate(): Promise<LabelingPromptTemplate> {
  return fetchAPI<LabelingPromptTemplate>('/labeling-prompt-templates/default');
}

/**
 * Create a new labeling prompt template
 */
export async function createLabelingPromptTemplate(
  template: LabelingPromptTemplateCreate
): Promise<LabelingPromptTemplate> {
  return fetchAPI<LabelingPromptTemplate>('/labeling-prompt-templates', {
    method: 'POST',
    body: JSON.stringify(template),
  });
}

/**
 * Update an existing labeling prompt template
 */
export async function updateLabelingPromptTemplate(
  id: string,
  updates: LabelingPromptTemplateUpdate
): Promise<LabelingPromptTemplate> {
  return fetchAPI<LabelingPromptTemplate>(`/labeling-prompt-templates/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(updates),
  });
}

/**
 * Delete a labeling prompt template
 */
export async function deleteLabelingPromptTemplate(
  id: string
): Promise<LabelingPromptTemplateDeleteResponse> {
  return fetchAPI<LabelingPromptTemplateDeleteResponse>(
    `/labeling-prompt-templates/${id}`,
    {
      method: 'DELETE',
    }
  );
}

/**
 * Set a template as the default
 */
export async function setDefaultLabelingPromptTemplate(
  id: string
): Promise<LabelingPromptTemplateSetDefaultResponse> {
  return fetchAPI<LabelingPromptTemplateSetDefaultResponse>(
    `/labeling-prompt-templates/${id}/set-default`,
    {
      method: 'POST',
    }
  );
}

/**
 * Get the number of labeling jobs using a specific template
 */
export async function getLabelingPromptTemplateUsageCount(
  id: string
): Promise<{ template_id: string; usage_count: number }> {
  return fetchAPI<{ template_id: string; usage_count: number }>(
    `/labeling-prompt-templates/${id}/usage-count`
  );
}

/**
 * Export labeling prompt templates to JSON format
 *
 * @param templateIds - Optional array of template IDs to export. If not provided, exports all custom templates.
 */
export async function exportLabelingPromptTemplates(
  templateIds?: string[]
): Promise<{
  version: string;
  exported_at: string;
  templates: Array<{
    name: string;
    description: string | null;
    system_message: string;
    user_prompt_template: string;
    temperature: number;
    max_tokens: number;
    top_p: number;
    is_default: boolean;
  }>;
}> {
  return fetchAPI(
    '/labeling-prompt-templates/export',
    {
      method: 'POST',
      body: JSON.stringify(templateIds || null),
    }
  );
}

/**
 * Import labeling prompt templates from JSON format
 *
 * @param importData - The export data containing templates
 * @param overwriteDuplicates - Whether to overwrite templates with the same name (default: false)
 */
export async function importLabelingPromptTemplates(
  importData: {
    version: string;
    exported_at: string;
    templates: Array<{
      name: string;
      description: string | null;
      system_message: string;
      user_prompt_template: string;
      temperature: number;
      max_tokens: number;
      top_p: number;
      is_default: boolean;
    }>;
  },
  overwriteDuplicates: boolean = false
): Promise<{
  success: boolean;
  message: string;
  imported_count: number;
  skipped_count: number;
  overwritten_count: number;
  failed_count: number;
  details: string[];
}> {
  return fetchAPI(
    '/labeling-prompt-templates/import',
    {
      method: 'POST',
      body: JSON.stringify({
        ...importData,
        overwrite_duplicates: overwriteDuplicates,
      }),
    }
  );
}
