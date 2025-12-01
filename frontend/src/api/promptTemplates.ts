/**
 * API client for prompt template operations.
 *
 * This module provides functions to interact with the backend prompt templates API.
 */

import {
  PromptTemplate,
  PromptTemplateCreate,
  PromptTemplateUpdate,
  PromptTemplateListResponse,
  PromptTemplateListParams,
  PromptTemplateExport,
  PromptTemplateImportResult,
} from '../types/promptTemplate';
import { fetchAPI, buildQueryString } from './client';

/**
 * Get list of prompt templates with optional filters
 */
export async function getPromptTemplates(
  params?: PromptTemplateListParams
): Promise<PromptTemplateListResponse> {
  const query = params ? buildQueryString(params) : '';
  const endpoint = `/prompt-templates${query ? `?${query}` : ''}`;
  return fetchAPI<PromptTemplateListResponse>(endpoint);
}

/**
 * Get a single prompt template by ID
 */
export async function getPromptTemplate(
  id: string
): Promise<PromptTemplate> {
  return fetchAPI<PromptTemplate>(`/prompt-templates/${id}`);
}

/**
 * Create a new prompt template
 */
export async function createPromptTemplate(
  template: PromptTemplateCreate
): Promise<PromptTemplate> {
  return fetchAPI<PromptTemplate>('/prompt-templates', {
    method: 'POST',
    body: JSON.stringify(template),
  });
}

/**
 * Update an existing prompt template
 */
export async function updatePromptTemplate(
  id: string,
  updates: PromptTemplateUpdate
): Promise<PromptTemplate> {
  return fetchAPI<PromptTemplate>(`/prompt-templates/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(updates),
  });
}

/**
 * Delete a prompt template
 */
export async function deletePromptTemplate(id: string): Promise<void> {
  return fetchAPI<void>(`/prompt-templates/${id}`, {
    method: 'DELETE',
  });
}

/**
 * Toggle the favorite status of a prompt template
 */
export async function togglePromptTemplateFavorite(
  id: string
): Promise<PromptTemplate> {
  return fetchAPI<PromptTemplate>(
    `/prompt-templates/${id}/favorite`,
    {
      method: 'POST',
    }
  );
}

/**
 * Duplicate a prompt template
 */
export async function duplicatePromptTemplate(
  id: string
): Promise<PromptTemplate> {
  return fetchAPI<PromptTemplate>(
    `/prompt-templates/${id}/duplicate`,
    {
      method: 'POST',
    }
  );
}

/**
 * Get list of favorite prompt templates
 */
export async function getFavoritePromptTemplates(params?: {
  page?: number;
  limit?: number;
}): Promise<PromptTemplateListResponse> {
  const query = params ? buildQueryString(params) : '';
  const endpoint = `/prompt-templates/favorites${query ? `?${query}` : ''}`;
  return fetchAPI<PromptTemplateListResponse>(endpoint);
}

/**
 * Export prompt templates to JSON format
 *
 * @param templateIds - Optional array of template IDs to export. If not provided, exports all templates.
 * @returns Export data with templates array and metadata
 */
export async function exportPromptTemplates(
  templateIds?: string[]
): Promise<PromptTemplateExport> {
  return fetchAPI<PromptTemplateExport>('/prompt-templates/export', {
    method: 'POST',
    body: JSON.stringify(templateIds || null),
  });
}

/**
 * Import prompt templates from JSON format
 *
 * @param importData - JSON data containing templates to import
 * @param overwriteDuplicates - Whether to overwrite templates with same name (default: false)
 * @returns Import result with counts of created/updated/skipped templates
 */
export async function importPromptTemplates(
  importData: PromptTemplateExport,
  overwriteDuplicates: boolean = false
): Promise<PromptTemplateImportResult> {
  return fetchAPI<PromptTemplateImportResult>('/prompt-templates/import', {
    method: 'POST',
    body: JSON.stringify({
      ...importData,
      overwrite_duplicates: overwriteDuplicates,
    }),
  });
}
