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

/** What a batch request did, feature by feature (004 batch enhanced labeling). */
export interface EnhancedLabelingBatchResponse {
  extraction_id: string;
  queued: string[];
  already_active: string[];
  skipped: { feature_id: string; reason: string }[];
  considered: number;
}

/**
 * Queue enhanced labeling for a SUBSET of one extraction's features: the explicit
 * `feature_ids`, or (`starred_only`) every feature starred by hand and not yet
 * enhanced-labeled. Bounded server-side; a feature already in flight is reported,
 * never queued twice.
 */
export async function startEnhancedLabelingBatch(
  extractionId: string,
  body: { feature_ids?: string[]; starred_only?: boolean }
): Promise<EnhancedLabelingBatchResponse> {
  return fetchAPI<EnhancedLabelingBatchResponse>(
    `/extractions/${extractionId}/label/enhanced/batch`,
    { method: 'POST', body: JSON.stringify(body) }
  );
}
