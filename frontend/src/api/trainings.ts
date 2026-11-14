/**
 * Training API client functions.
 *
 * Provides functions for fetching training data from the backend API.
 */

import { fetchAPI, buildQueryString } from './client';
import type { TrainingMetric } from '../types/training';

/**
 * Training metric data point returned from the API: the one definition in
 * `types/training.ts`, which mirrors the backend's `TrainingMetricResponse`.
 *
 * This file used to declare its own copy, and it drifted (review R2-D, R2D-9): it
 * typed `id` as a string (the backend sends an int) and `fvu`, `l0_sparsity`,
 * `dead_neurons` and `learning_rate` as never null (each is Optional).
 */
export type { TrainingMetric };

/**
 * Response from the training metrics endpoint.
 */
export interface TrainingMetricsResponse {
  data: TrainingMetric[];
}

/**
 * Options for fetching training metrics.
 */
export interface FetchMetricsOptions {
  /** Start step (inclusive) */
  start_step?: number;
  /** End step (inclusive) */
  end_step?: number;
  /** Maximum number of metrics to return (default: 1000) */
  limit?: number;
  /**
   * Only the aggregate rows (layer_idx null), one per logged step. A raw window is shared by
   * one row per SAE and one held-out row per SAE for every hook type (review R1-A, A5).
   */
  aggregate_only?: boolean;
}

/**
 * Fetch training metrics for a specific training job.
 *
 * @param trainingId - The training job ID
 * @param options - Optional parameters for filtering/limiting results
 * @returns Array of training metrics sorted by step
 *
 * @example
 * ```typescript
 * // Get last 20 metrics
 * const metrics = await fetchTrainingMetrics('training_123', { limit: 20 });
 * ```
 */
export async function fetchTrainingMetrics(
  trainingId: string,
  options: FetchMetricsOptions = {}
): Promise<TrainingMetric[]> {
  const queryString = buildQueryString(options);
  const endpoint = `/trainings/${trainingId}/metrics${queryString ? `?${queryString}` : ''}`;

  const response = await fetchAPI<TrainingMetricsResponse>(endpoint);
  return response.data;
}
