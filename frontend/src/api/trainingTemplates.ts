/**
 * API client for training template operations.
 *
 * This module provides functions to interact with the backend training templates API.
 */

import {
  TrainingTemplate,
  TrainingTemplateCreate,
  TrainingTemplateUpdate,
  TrainingTemplateListResponse,
  TrainingTemplateListParams,
  TrainingTemplateExport,
  TrainingTemplateImportResult,
} from '../types/trainingTemplate';
import { fetchAPI, buildQueryString } from './client';

/**
 * Get list of training templates with optional filters
 */
export async function getTrainingTemplates(
  params?: TrainingTemplateListParams
): Promise<TrainingTemplateListResponse> {
  const query = params ? buildQueryString(params) : '';
  const endpoint = `/training-templates${query ? `?${query}` : ''}`;
  return fetchAPI<TrainingTemplateListResponse>(endpoint);
}

/**
 * Get a single training template by ID
 */
export async function getTrainingTemplate(
  id: string
): Promise<TrainingTemplate> {
  return fetchAPI<TrainingTemplate>(`/training-templates/${id}`);
}

/**
 * Create a new training template
 */
export async function createTrainingTemplate(
  template: TrainingTemplateCreate
): Promise<TrainingTemplate> {
  return fetchAPI<TrainingTemplate>('/training-templates', {
    method: 'POST',
    body: JSON.stringify(template),
  });
}

/**
 * Update an existing training template
 */
export async function updateTrainingTemplate(
  id: string,
  updates: TrainingTemplateUpdate
): Promise<TrainingTemplate> {
  return fetchAPI<TrainingTemplate>(`/training-templates/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(updates),
  });
}

/**
 * Delete a training template
 */
export async function deleteTrainingTemplate(id: string): Promise<void> {
  return fetchAPI<void>(`/training-templates/${id}`, {
    method: 'DELETE',
  });
}

/**
 * Toggle the favorite status of a training template
 */
export async function toggleTrainingTemplateFavorite(
  id: string
): Promise<TrainingTemplate> {
  return fetchAPI<TrainingTemplate>(
    `/training-templates/${id}/favorite`,
    {
      method: 'POST',
    }
  );
}

/**
 * Get list of favorite training templates
 */
export async function getFavoriteTrainingTemplates(params?: {
  page?: number;
  limit?: number;
}): Promise<TrainingTemplateListResponse> {
  const query = params ? buildQueryString(params) : '';
  const endpoint = `/training-templates/favorites${query ? `?${query}` : ''}`;
  return fetchAPI<TrainingTemplateListResponse>(endpoint);
}

/**
 * Export training templates to JSON format
 *
 * @param templateIds - Optional array of template IDs to export. If not provided, exports all templates.
 * @returns Export data with templates array and metadata
 */
export async function exportTrainingTemplates(
  templateIds?: string[]
): Promise<TrainingTemplateExport> {
  return fetchAPI<TrainingTemplateExport>('/training-templates/export', {
    method: 'POST',
    body: JSON.stringify(templateIds || null),
  });
}

/**
 * Import training templates from JSON format
 *
 * @param importData - JSON data containing templates to import
 * @param overwriteDuplicates - Whether to overwrite templates with same name (default: false)
 * @returns Import result with counts of created/updated/skipped templates
 */
export async function importTrainingTemplates(
  importData: TrainingTemplateExport,
  overwriteDuplicates: boolean = false
): Promise<TrainingTemplateImportResult> {
  return fetchAPI<TrainingTemplateImportResult>('/training-templates/import', {
    method: 'POST',
    body: JSON.stringify({
      ...importData,
      overwrite_duplicates: overwriteDuplicates,
    }),
  });
}
