/**
 * API client for labeling operations.
 *
 * This module provides functions to interact with the backend labeling API for
 * independent semantic labeling of extracted SAE features.
 */

import {
  LabelingJob,
  LabelingConfigRequest,
  LabelingListResponse,
  LabelingListParams,
} from '../types/labeling';
import { fetchAPI, buildQueryString } from './client';

/**
 * Start a semantic labeling job for a completed extraction.
 *
 * This creates a labeling job and queues it for async processing. Features
 * are labeled independently from extraction, allowing re-labeling without
 * re-extraction.
 *
 * @param config - Labeling configuration (extraction_job_id, labeling_method, etc.)
 * @returns Promise resolving to the created LabelingJob
 * @throws {APIError} If extraction not found, active labeling exists, or validation fails
 *
 * @example
 * ```typescript
 * const labelingJob = await startLabeling({
 *   extraction_job_id: 'extr_20251107_020805_train_a0',
 *   labeling_method: LabelingMethod.OPENAI,
 *   openai_model: 'gpt-4o-mini',
 *   batch_size: 10
 * });
 * ```
 */
export async function startLabeling(
  config: LabelingConfigRequest
): Promise<LabelingJob> {
  return fetchAPI<LabelingJob>('/labeling', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

/**
 * Get the status of a specific labeling job.
 *
 * @param labelingJobId - ID of the labeling job
 * @returns Promise resolving to the LabelingJob with status, progress, and statistics
 * @throws {APIError} If labeling job not found
 *
 * @example
 * ```typescript
 * const labelingJob = await getLabelingJob('label_extr_20251107_020805_train_a0_20251108_123456');
 * console.log(`Progress: ${labelingJob.progress * 100}%`);
 * ```
 */
export async function getLabelingJob(
  labelingJobId: string
): Promise<LabelingJob> {
  return fetchAPI<LabelingJob>(`/labeling/${labelingJobId}`);
}

/**
 * Get a paginated list of labeling jobs with optional filtering.
 *
 * @param params - Optional query parameters (extraction_job_id, limit, offset)
 * @returns Promise resolving to LabelingListResponse with jobs and metadata
 *
 * @example
 * ```typescript
 * // Get all labeling jobs
 * const response = await listLabelingJobs();
 *
 * // Get labeling jobs for specific extraction
 * const response = await listLabelingJobs({
 *   extraction_job_id: 'extr_20251107_020805_train_a0',
 *   limit: 50,
 *   offset: 0
 * });
 * ```
 */
export async function listLabelingJobs(
  params?: LabelingListParams
): Promise<LabelingListResponse> {
  const query = params ? buildQueryString(params) : '';
  const endpoint = `/labeling${query ? `?${query}` : ''}`;
  return fetchAPI<LabelingListResponse>(endpoint);
}

/**
 * Cancel an active labeling job.
 *
 * Only jobs with status "queued" or "labeling" can be cancelled.
 *
 * @param labelingJobId - ID of the labeling job to cancel
 * @returns Promise resolving to success message
 * @throws {APIError} If job not found or not in cancellable state
 *
 * @example
 * ```typescript
 * await cancelLabeling('label_extr_20251107_020805_train_a0_20251108_123456');
 * console.log('Labeling job cancelled successfully');
 * ```
 */
export async function cancelLabeling(
  labelingJobId: string
): Promise<{ message: string }> {
  return fetchAPI<{ message: string }>(`/labeling/${labelingJobId}/cancel`, {
    method: 'POST',
  });
}

/**
 * Delete a labeling job record.
 *
 * This does NOT delete the features or their labels, only the labeling job
 * record itself. Feature labels will remain intact.
 *
 * Active labeling jobs (queued/labeling) must be cancelled before deletion.
 *
 * @param labelingJobId - ID of the labeling job to delete
 * @returns Promise resolving to void (204 No Content)
 * @throws {APIError} If job not found or is still active
 *
 * @example
 * ```typescript
 * await deleteLabeling('label_extr_20251107_020805_train_a0_20251108_123456');
 * console.log('Labeling job deleted successfully');
 * ```
 */
export async function deleteLabeling(labelingJobId: string): Promise<void> {
  return fetchAPI<void>(`/labeling/${labelingJobId}`, {
    method: 'DELETE',
  });
}

/**
 * Convenience function to start labeling for an extraction.
 *
 * This is a shorthand for calling startLabeling() with the extraction_job_id
 * already set in the config.
 *
 * @param extractionId - ID of the extraction to label
 * @param config - Labeling configuration (labeling_method, openai_model, etc.)
 * @returns Promise resolving to the created LabelingJob
 * @throws {APIError} If extraction not found, active labeling exists, or validation fails
 *
 * @example
 * ```typescript
 * const labelingJob = await labelExtraction('extr_20251107_020805_train_a0', {
 *   extraction_job_id: 'extr_20251107_020805_train_a0',  // Will be overridden
 *   labeling_method: LabelingMethod.OPENAI,
 *   openai_model: 'gpt-4o-mini'
 * });
 * ```
 */
export async function labelExtraction(
  extractionId: string,
  config: LabelingConfigRequest
): Promise<LabelingJob> {
  return fetchAPI<LabelingJob>(`/extractions/${extractionId}/label`, {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

/**
 * Get labeling jobs for a specific extraction.
 *
 * Helper function that filters labeling jobs by extraction_job_id.
 *
 * @param extractionJobId - ID of the extraction to get labeling jobs for
 * @param limit - Maximum number of results (default: 50)
 * @param offset - Number of results to skip (default: 0)
 * @returns Promise resolving to LabelingListResponse
 *
 * @example
 * ```typescript
 * const response = await getLabelingJobsForExtraction('extr_20251107_020805_train_a0');
 * console.log(`Found ${response.meta.total} labeling jobs`);
 * ```
 */
export async function getLabelingJobsForExtraction(
  extractionJobId: string,
  limit: number = 50,
  offset: number = 0
): Promise<LabelingListResponse> {
  return listLabelingJobs({
    extraction_job_id: extractionJobId,
    limit,
    offset,
  });
}
