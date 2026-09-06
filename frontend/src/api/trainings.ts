/**
 * Training API client functions.
 *
 * Provides functions for fetching training data from the backend API.
 */

import { fetchAPI, buildQueryString } from './client';

/**
 * Training metric data point returned from the API.
 */
export interface TrainingMetric {
  id: string;
  training_id: string;
  step: number;
  loss: number;
  l0_sparsity?: number;
  dead_neurons?: number;
  learning_rate?: number;
  fvu?: number;
  timestamp: string;
}

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
