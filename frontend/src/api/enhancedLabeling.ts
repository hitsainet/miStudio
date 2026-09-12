/**
 * API client for enhanced per-feature two-pass labeling.
 */

import { EnhancedLabelingJob } from '../types/enhancedLabeling';
import { fetchAPI } from './client';

/**
 * Start an enhanced two-pass labeling job for a single feature.
 * If a job is already active for this feature, returns the existing one (HTTP 200).
 */
export async function startEnhancedLabeling(featureId: string): Promise<EnhancedLabelingJob> {
  return fetchAPI<EnhancedLabelingJob>(
    `/features/${featureId}/label/enhanced`,
    { method: 'POST' }
  );
}

/**
 * Get the most recent enhanced labeling job for a feature, or null if none exists.
 */
export async function getLatestEnhancedLabelingJob(
  featureId: string
): Promise<EnhancedLabelingJob | null> {
  return fetchAPI<EnhancedLabelingJob | null>(
    `/features/${featureId}/label/enhanced/latest`
  );
}
