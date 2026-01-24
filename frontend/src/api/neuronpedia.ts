/**
 * API client for Neuronpedia Export operations.
 *
 * This module provides functions to interact with the backend Neuronpedia API.
 * All functions make REAL HTTP requests to the production backend.
 */

import {
  NeuronpediaExportConfig,
  NeuronpediaExportJobResponse,
  NeuronpediaExportJob,
  NeuronpediaExportJobListResponse,
  ComputeDashboardDataRequest,
  ComputeDashboardDataResponse,
  FeatureDashboardData,
} from '../types/neuronpedia';
import { fetchAPI, buildQueryString } from './client';

/**
 * Start a Neuronpedia export job.
 */
export async function startExport(
  saeId: string,
  config: NeuronpediaExportConfig
): Promise<NeuronpediaExportJobResponse> {
  // Convert camelCase to snake_case for backend
  const requestBody = {
    sae_id: saeId,
    config: {
      feature_selection: config.featureSelection,
      feature_indices: config.featureIndices,
      include_logit_lens: config.includeLogitLens,
      logit_lens_k: config.logitLensK,
      include_histograms: config.includeHistograms,
      histogram_bins: config.histogramBins,
      include_top_tokens: config.includeTopTokens,
      top_tokens_k: config.topTokensK,
      include_saelens_format: config.includeSaelensFormat,
      include_explanations: config.includeExplanations,
    },
  };

  const response = await fetchAPI<{
    job_id: string;
    status: string;
    message: string;
  }>('/neuronpedia/export', {
    method: 'POST',
    body: JSON.stringify(requestBody),
  });

  return {
    jobId: response.job_id,
    status: response.status as NeuronpediaExportJobResponse['status'],
    message: response.message,
  };
}

/**
 * Get status of an export job.
 */
export async function getExportStatus(jobId: string): Promise<NeuronpediaExportJob> {
  const response = await fetchAPI<{
    id: string;
    sae_id: string;
    status: string;
    progress: number;
    current_stage?: string;
    feature_count?: number;
    created_at: string;
    started_at?: string;
    completed_at?: string;
    output_path?: string;
    file_size_bytes?: number;
    error_message?: string;
    download_url?: string;
  }>(`/neuronpedia/export/${jobId}`);

  return {
    id: response.id,
    saeId: response.sae_id,
    status: response.status as NeuronpediaExportJob['status'],
    progress: response.progress,
    currentStage: response.current_stage,
    featureCount: response.feature_count,
    createdAt: response.created_at,
    startedAt: response.started_at,
    completedAt: response.completed_at,
    outputPath: response.output_path,
    fileSizeBytes: response.file_size_bytes,
    errorMessage: response.error_message,
    downloadUrl: response.download_url,
  };
}

/**
 * Download export archive.
 * Returns a Blob that can be saved as a file.
 */
export async function downloadExport(jobId: string): Promise<Blob> {
  const response = await fetch(`/api/v1/neuronpedia/export/${jobId}/download`, {
    method: 'GET',
    credentials: 'include',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Download failed' }));
    throw new Error(error.detail || `Download failed: ${response.status}`);
  }

  return response.blob();
}

/**
 * Cancel an in-progress export job.
 */
export async function cancelExport(jobId: string): Promise<{ message: string }> {
  return fetchAPI<{ message: string }>(`/neuronpedia/export/${jobId}/cancel`, {
    method: 'POST',
  });
}

/**
 * Delete an export job and its files.
 */
export async function deleteExport(jobId: string): Promise<{ message: string }> {
  return fetchAPI<{ message: string }>(`/neuronpedia/export/${jobId}`, {
    method: 'DELETE',
  });
}

/**
 * List export jobs with optional filters.
 */
export async function listExports(params?: {
  saeId?: string;
  status?: string;
  skip?: number;
  limit?: number;
}): Promise<NeuronpediaExportJobListResponse> {
  const queryParams: Record<string, string | number | undefined> = {
    sae_id: params?.saeId,
    status: params?.status,
    skip: params?.skip,
    limit: params?.limit,
  };

  const query = buildQueryString(queryParams);
  const endpoint = `/neuronpedia/exports${query ? `?${query}` : ''}`;

  const response = await fetchAPI<{
    jobs: Array<{
      id: string;
      sae_id: string;
      status: string;
      progress: number;
      current_stage?: string;
      feature_count?: number;
      created_at: string;
      started_at?: string;
      completed_at?: string;
      output_path?: string;
      file_size_bytes?: number;
      error_message?: string;
      download_url?: string;
    }>;
    total: number;
  }>(endpoint);

  return {
    jobs: response.jobs.map((job) => ({
      id: job.id,
      saeId: job.sae_id,
      status: job.status as NeuronpediaExportJob['status'],
      progress: job.progress,
      currentStage: job.current_stage,
      featureCount: job.feature_count,
      createdAt: job.created_at,
      startedAt: job.started_at,
      completedAt: job.completed_at,
      outputPath: job.output_path,
      fileSizeBytes: job.file_size_bytes,
      errorMessage: job.error_message,
      downloadUrl: job.download_url,
    })),
    total: response.total,
  };
}

/**
 * Compute dashboard data for features on-demand.
 */
export async function computeDashboardData(
  request: ComputeDashboardDataRequest
): Promise<ComputeDashboardDataResponse> {
  const requestBody = {
    sae_id: request.saeId,
    feature_indices: request.featureIndices,
    include_logit_lens: request.includeLogitLens,
    include_histograms: request.includeHistograms,
    include_top_tokens: request.includeTopTokens,
    force_recompute: request.forceRecompute,
  };

  const response = await fetchAPI<{
    features_computed: number;
    status: string;
    message: string;
  }>('/neuronpedia/compute-dashboard-data', {
    method: 'POST',
    body: JSON.stringify(requestBody),
  });

  return {
    featuresComputed: response.features_computed,
    status: response.status as ComputeDashboardDataResponse['status'],
    message: response.message,
  };
}

/**
 * Get dashboard data for a single feature.
 */
export async function getFeatureDashboardData(
  saeId: string,
  featureIndex: number
): Promise<FeatureDashboardData | null> {
  try {
    const response = await fetchAPI<{
      feature_id: string;
      logit_lens_data?: {
        top_positive: Array<{ token: string; value: number }>;
        top_negative: Array<{ token: string; value: number }>;
      };
      histogram_data?: {
        bin_edges: number[];
        counts: number[];
        total_count: number;
        nonzero_count: number;
        mean?: number;
        std?: number;
        min?: number;
        max?: number;
      };
      top_tokens?: Array<{
        token: string;
        total_activation: number;
        count: number;
        mean_activation: number;
        max_activation: number;
      }>;
      computed_at?: string;
    }>(`/neuronpedia/sae/${saeId}/feature/${featureIndex}/dashboard`);

    return {
      featureId: response.feature_id,
      logitLensData: response.logit_lens_data
        ? {
            topPositive: response.logit_lens_data.top_positive.map((t) => ({
              token: t.token,
              value: t.value,
            })),
            topNegative: response.logit_lens_data.top_negative.map((t) => ({
              token: t.token,
              value: t.value,
            })),
          }
        : undefined,
      histogramData: response.histogram_data
        ? {
            binEdges: response.histogram_data.bin_edges,
            counts: response.histogram_data.counts,
            totalCount: response.histogram_data.total_count,
            nonzeroCount: response.histogram_data.nonzero_count,
            mean: response.histogram_data.mean,
            std: response.histogram_data.std,
            min: response.histogram_data.min,
            max: response.histogram_data.max,
          }
        : undefined,
      topTokens: response.top_tokens?.map((t) => ({
        token: t.token,
        totalActivation: t.total_activation,
        count: t.count,
        meanActivation: t.mean_activation,
        maxActivation: t.max_activation,
      })),
      computedAt: response.computed_at,
    };
  } catch {
    return null;
  }
}

/**
 * Helper function to trigger file download in browser.
 */
export function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Format file size for display.
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

/**
 * Format duration for display.
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  if (minutes < 60) return `${minutes}m ${remainingSeconds}s`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return `${hours}h ${remainingMinutes}m`;
}

// ============================================================================
// Local Neuronpedia Push API
// ============================================================================

export interface LocalPushConfig {
  includeActivations: boolean;
  includeExplanations: boolean;
  maxActivationsPerFeature: number;
  featureIndices?: number[];
}

export interface LocalPushResult {
  success: boolean;
  modelId: string;
  sourceId: string;
  neuronsCreated: number;
  activationsCreated: number;
  explanationsCreated: number;
  neuronpediaUrl?: string;
}

export interface LocalNeuronpediaStatus {
  configured: boolean;
  dbUrlSet: boolean;
  publicUrl?: string;
  connected: boolean;
  error?: string;
}

/**
 * Check local Neuronpedia connection status.
 */
export async function getLocalStatus(): Promise<LocalNeuronpediaStatus> {
  const response = await fetchAPI<{
    configured: boolean;
    db_url_set: boolean;
    public_url?: string;
    connected: boolean;
    error?: string;
  }>('/neuronpedia/local-status');

  return {
    configured: response.configured,
    dbUrlSet: response.db_url_set,
    publicUrl: response.public_url,
    connected: response.connected,
    error: response.error,
  };
}

/**
 * Push SAE features to local Neuronpedia instance.
 */
export async function pushToLocal(
  saeId: string,
  config: LocalPushConfig
): Promise<LocalPushResult> {
  const queryParams = new URLSearchParams({
    sae_id: saeId,
    include_activations: String(config.includeActivations),
    include_explanations: String(config.includeExplanations),
    max_activations_per_feature: String(config.maxActivationsPerFeature),
  });

  if (config.featureIndices && config.featureIndices.length > 0) {
    config.featureIndices.forEach((idx) => {
      queryParams.append('feature_indices', String(idx));
    });
  }

  const response = await fetchAPI<{
    success: boolean;
    model_id: string;
    source_id: string;
    neurons_created: number;
    activations_created: number;
    explanations_created: number;
    neuronpedia_url?: string;
  }>(`/neuronpedia/push-local?${queryParams.toString()}`, {
    method: 'POST',
  });

  return {
    success: response.success,
    modelId: response.model_id,
    sourceId: response.source_id,
    neuronsCreated: response.neurons_created,
    activationsCreated: response.activations_created,
    explanationsCreated: response.explanations_created,
    neuronpediaUrl: response.neuronpedia_url,
  };
}
