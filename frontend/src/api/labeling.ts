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
  LabelingCoverage,
  LabelingResumeSweep,
  AvailableJudges,
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

/**
 * Ask what still needs labeling in an extraction.
 *
 * @param extractionJobId - the extraction to report on
 * @param options.promptFingerprint - with `judgeModel`, also counts and offers
 *   features adjudicated by a DIFFERENT judge. Omit for a routine resume, which
 *   must never revisit a verdict.
 * @param options.resumeLimit - how many ids to return, capped at 2000 by the
 *   panel route's own limit.
 */
export async function getLabelingCoverage(
  extractionJobId: string,
  options: {
    promptFingerprint?: string;
    judgeModel?: string;
    resumeLimit?: number;
    maxAttempts?: number;
    /**
     * Draw the batch from ONE outcome. A sample must ask a single question:
     * "do the failures still fail?" and "does the judge work?" are different,
     * and a mixed batch of 20 answers neither.
     */
    only?: 'failed' | 'pending';
  } = {}
): Promise<LabelingCoverage> {
  const params = new URLSearchParams();
  if (options.promptFingerprint) params.set('prompt_fingerprint', options.promptFingerprint);
  if (options.judgeModel) params.set('judge_model', options.judgeModel);
  if (options.resumeLimit !== undefined) params.set('resume_limit', String(options.resumeLimit));
  if (options.maxAttempts !== undefined) params.set('max_attempts', String(options.maxAttempts));
  if (options.only) params.set('only', options.only);
  const query = params.toString();
  return fetchAPI<LabelingCoverage>(
    `/labeling/${encodeURIComponent(extractionJobId)}/coverage${query ? `?${query}` : ''}`
  );
}

/**
 * Label ONLY the listed features, writing labels to the feature rows.
 *
 * This is how a resume runs. The backend route has existed since Feature 30 and
 * the frontend has simply never called it — which is why "resume" needed no new
 * endpoint. What was missing was never the ability to label a subset, it was the
 * ability to know WHICH subset.
 *
 * At most 2000 ids: Celery's soft limit is 10 h and labeling measures ~8 s per
 * feature. Larger sets must be split across several jobs.
 */
export async function startLabelingPanel(
  config: LabelingConfigRequest & { feature_ids: string[] }
): Promise<LabelingJob> {
  return fetchAPI<LabelingJob>('/labeling/panel', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

/**
 * Start a multi-batch resume.
 *
 * `maxBatches` is REQUIRED by the signature and by the server. An open-ended
 * sweep is a request to spend an unknown number of GPU-hours, and it must not be
 * startable by omitting an argument.
 */
export async function startResumeSweep(
  extractionJobId: string,
  maxBatches: number,
  /*
   * The config is FROZEN on the sweep row, so whatever predicate travels here
   * is the predicate every batch selects with — even if someone edits the
   * template while the sweep runs. `max_batches` is computed from a coverage
   * read that carries the fingerprint and judge; selecting without them made
   * the sweep run a set it was not sized for.
   */
  config: LabelingConfigRequest & {
    prompt_fingerprint?: string;
    judge_model?: string;
  },
  batchSize = 2000
): Promise<LabelingResumeSweep> {
  return fetchAPI<LabelingResumeSweep>(
    `/labeling/${encodeURIComponent(extractionJobId)}/resume-sweep`,
    {
      method: 'POST',
      body: JSON.stringify({ max_batches: maxBatches, config, batch_size: batchSize }),
    }
  );
}

/** How far a sweep has got. */
export async function getResumeSweep(sweepId: string): Promise<LabelingResumeSweep> {
  return fetchAPI<LabelingResumeSweep>(`/labeling/resume-sweeps/${encodeURIComponent(sweepId)}`);
}

/** Stop a sweep after its current batch. Cooperative — nothing is killed. */
export async function cancelResumeSweep(sweepId: string): Promise<LabelingResumeSweep> {
  return fetchAPI<LabelingResumeSweep>(
    `/labeling/resume-sweeps/${encodeURIComponent(sweepId)}/cancel`,
    { method: 'POST' }
  );
}

/**
 * Which models this job's labeling endpoint can serve now.
 *
 * Keyed on the job, so the endpoint comes from the stored row rather than from
 * the caller. Used by resume to offer a different judge when the original is
 * no longer available — the case the whole feature exists for.
 */
export async function getAvailableJudges(labelingJobId: string): Promise<AvailableJudges> {
  return fetchAPI<AvailableJudges>(
    `/labeling/${encodeURIComponent(labelingJobId)}/available-judges`
  );
}
