/**
 * API client for extraction template operations.
 *
 * This module provides functions to interact with the backend extraction templates API.
 */

import {
  ExtractionTemplate,
  ExtractionTemplateCreate,
  ExtractionTemplateUpdate,
  ExtractionTemplateListResponse,
  ExtractionTemplateListParams,
  ExtractionTemplateExport,
  ExtractionTemplateImportResult,
} from '../types/extractionTemplate';
import { fetchAPI, buildQueryString } from './client';

/**
 * Get list of extraction templates with optional filters
 */
export async function getExtractionTemplates(
  params?: ExtractionTemplateListParams
): Promise<ExtractionTemplateListResponse> {
  const query = params ? buildQueryString(params) : '';
  const endpoint = `/extraction-templates${query ? `?${query}` : ''}`;
  return fetchAPI<ExtractionTemplateListResponse>(endpoint);
}

/**
 * Get a single extraction template by ID
 */
export async function getExtractionTemplate(
  id: string
): Promise<ExtractionTemplate> {
  return fetchAPI<ExtractionTemplate>(`/extraction-templates/${id}`);
}

/**
 * Create a new extraction template
 */
export async function createExtractionTemplate(
  template: ExtractionTemplateCreate
): Promise<ExtractionTemplate> {
  return fetchAPI<ExtractionTemplate>('/extraction-templates', {
    method: 'POST',
    body: JSON.stringify(template),
  });
}

/**
 * Update an existing extraction template
 */
export async function updateExtractionTemplate(
  id: string,
  updates: ExtractionTemplateUpdate
): Promise<ExtractionTemplate> {
  return fetchAPI<ExtractionTemplate>(`/extraction-templates/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(updates),
  });
}

/**
 * Delete an extraction template
 */
export async function deleteExtractionTemplate(id: string): Promise<void> {
  return fetchAPI<void>(`/extraction-templates/${id}`, {
    method: 'DELETE',
  });
}

/**
 * Toggle the favorite status of an extraction template
 */
export async function toggleExtractionTemplateFavorite(
  id: string
): Promise<ExtractionTemplate> {
  return fetchAPI<ExtractionTemplate>(
    `/extraction-templates/${id}/favorite`,
    {
      method: 'POST',
    }
  );
}

/**
 * Get list of favorite extraction templates
 */
export async function getFavoriteExtractionTemplates(params?: {
  page?: number;
  limit?: number;
}): Promise<ExtractionTemplateListResponse> {
  const query = params ? buildQueryString(params) : '';
  const endpoint = `/extraction-templates/favorites${query ? `?${query}` : ''}`;
  return fetchAPI<ExtractionTemplateListResponse>(endpoint);
}

/**
 * Export extraction templates to JSON format
 *
 * @param templateIds - Optional array of template IDs to export. If not provided, exports all templates.
 */
export async function exportExtractionTemplates(
  templateIds?: string[]
): Promise<ExtractionTemplateExport> {
  return fetchAPI<ExtractionTemplateExport>(
    '/extraction-templates/export',
    {
      method: 'POST',
      body: JSON.stringify(templateIds || null),
    }
  );
}

/**
 * Import extraction templates from JSON format
 *
 * @param importData - The export data containing templates
 * @param overwriteDuplicates - Whether to overwrite templates with the same name (default: false)
 */
export async function importExtractionTemplates(
  importData: ExtractionTemplateExport,
  overwriteDuplicates: boolean = false
): Promise<ExtractionTemplateImportResult> {
  return fetchAPI<ExtractionTemplateImportResult>(
    '/extraction-templates/import',
    {
      method: 'POST',
      body: JSON.stringify({
        ...importData,
        overwrite_duplicates: overwriteDuplicates,
      }),
    }
  );
}
