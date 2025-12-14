/**
 * API client for model operations.
 *
 * This module provides functions to interact with the backend models API.
 * All functions make REAL HTTP requests to the production backend.
 */

import {
  Model,
  ModelDownloadRequest,
  ModelListResponse,
  ModelArchitecture,
  ActivationExtractionConfig,
  ActivationExtractionResult,
} from '../types/model';
import { fetchAPI, buildQueryString } from './client';

/**
 * Get list of models with optional filters
 */
export async function getModels(params?: {
  skip?: number;
  limit?: number;
  search?: string;
  architecture?: string;
  quantization?: string;
  status?: string;
  sort_by?: string;
  order?: 'asc' | 'desc';
}): Promise<ModelListResponse> {
  const query = params ? buildQueryString(params) : '';
  const endpoint = `/models${query ? `?${query}` : ''}`;
  return fetchAPI<ModelListResponse>(endpoint);
}

/**
 * Get a single model by ID
 */
export async function getModel(id: string): Promise<Model> {
  return fetchAPI<Model>(`/models/${id}`);
}

/**
 * Download a model from HuggingFace
 */
export async function downloadModel(
  request: ModelDownloadRequest
): Promise<Model> {
  return fetchAPI<Model>('/models/download', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Delete a model
 */
export async function deleteModel(id: string): Promise<void> {
  return fetchAPI<void>(`/models/${id}`, {
    method: 'DELETE',
  });
}

/**
 * Get model architecture details
 */
export async function getModelArchitecture(id: string): Promise<ModelArchitecture> {
  return fetchAPI<ModelArchitecture>(`/models/${id}/architecture`);
}

/**
 * Estimate resource requirements for an extraction job
 */
export async function estimateExtractionResources(
  modelId: string,
  config: ActivationExtractionConfig
): Promise<{
  model_id: string;
  dataset_id: string;
  estimates: {
    gpu_memory: {
      total_bytes: number;
      total_mb: number;
      total_gb: number;
      breakdown: Record<string, number>;
      warning: 'normal' | 'medium' | 'high';
    };
    disk_space: {
      total_bytes: number;
      total_mb: number;
      total_gb: number;
      per_layer_mb: number;
      warning: 'normal' | 'medium' | 'high';
    };
    processing_time: {
      total_seconds: number;
      time_str: string;
      estimated_batches: number;
      seconds_per_batch: number;
      warning: 'normal' | 'medium' | 'long';
    };
    warnings: string[];
    config_used: {
      hidden_size: number;
      num_layers: number;
      batch_size: number;
      max_samples: number;
      sequence_length: number;
    };
  };
}> {
  return fetchAPI(`/models/${modelId}/estimate-extraction`, {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

/**
 * Extract activations from a model
 */
export async function extractActivations(
  modelId: string,
  config: ActivationExtractionConfig
): Promise<ActivationExtractionResult> {
  console.log('[extractActivations] Sending request with config:', JSON.stringify(config, null, 2));
  return fetchAPI<ActivationExtractionResult>(`/models/${modelId}/extract-activations`, {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

/**
 * Update model metadata (PATCH)
 */
export async function updateModel(
  id: string,
  updates: Partial<Model>
): Promise<Model> {
  return fetchAPI<Model>(`/models/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(updates),
  });
}

/**
 * Cancel an in-progress model download
 */
export async function cancelModelDownload(id: string): Promise<{
  model_id: string;
  status: string;
  message: string;
}> {
  return fetchAPI<{
    model_id: string;
    status: string;
    message: string;
  }>(`/models/${id}/cancel`, {
    method: 'DELETE',
  });
}

/**
 * Get Celery task status by task ID
 */
export async function getTaskStatus(taskId: string): Promise<{
  task_id: string;
  status: string;
  result?: any;
  error?: string;
}> {
  return fetchAPI<{
    task_id: string;
    status: string;
    result?: any;
    error?: string;
  }>(`/models/tasks/${taskId}`);
}

/**
 * Get extraction history for a model
 */
export async function getModelExtractions(modelId: string): Promise<{
  model_id: string;
  model_name: string;
  extractions: any[];
  count: number;
}> {
  return fetchAPI<{
    model_id: string;
    model_name: string;
    extractions: any[];
    count: number;
  }>(`/models/${modelId}/extractions`);
}

/**
 * Cancel an in-progress extraction
 */
export async function cancelExtraction(
  modelId: string,
  extractionId: string
): Promise<{
  extraction_id: string;
  status: string;
  message: string;
}> {
  return fetchAPI<{
    extraction_id: string;
    status: string;
    message: string;
  }>(`/models/${modelId}/extractions/${extractionId}/cancel`, {
    method: 'POST',
  });
}

/**
 * Retry a failed extraction with optional parameter overrides
 */
export async function retryExtraction(
  modelId: string,
  extractionId: string,
  retryParams?: {
    batch_size?: number;
    max_samples?: number;
  }
): Promise<{
  original_extraction_id: string;
  new_extraction_id: string;
  job_id: string;
  status: string;
  message: string;
}> {
  return fetchAPI<{
    original_extraction_id: string;
    new_extraction_id: string;
    job_id: string;
    status: string;
    message: string;
  }>(`/models/${modelId}/extractions/${extractionId}/retry`, {
    method: 'POST',
    body: JSON.stringify(retryParams || {}),
  });
}

/**
 * Delete multiple extractions for a model
 */
export async function deleteExtractions(
  modelId: string,
  extractionIds: string[]
): Promise<{
  model_id: string;
  deleted_count: number;
  failed_count: number;
  deleted_ids: string[];
  failed_ids: string[];
  errors: Record<string, string>;
  message: string;
}> {
  return fetchAPI<{
    model_id: string;
    deleted_count: number;
    failed_count: number;
    deleted_ids: string[];
    failed_ids: string[];
    errors: Record<string, string>;
    message: string;
  }>(`/models/${modelId}/extractions`, {
    method: 'DELETE',
    body: JSON.stringify({ extraction_ids: extractionIds }),
  });
}

/**
 * Get list of locally cached HuggingFace models
 */
export async function getLocalModels(): Promise<{ models: string[] }> {
  return fetchAPI<{ models: string[] }>('/models/local-cache/list');
}

/**
 * Trigger NLP analysis for an extraction job.
 * This processes all features in the extraction to compute NLP analysis
 * (POS tags, NER, n-grams, clusters) and stores results persistently.
 *
 * @param extractionId - The extraction job ID to analyze
 * @param options - Optional configuration (featureIds to limit scope, batchSize, forceReprocess)
 * @returns Task status with task_id for progress tracking via WebSocket
 */
export async function triggerNlpAnalysis(
  extractionId: string,
  options?: {
    feature_ids?: string[];
    batch_size?: number;
    force_reprocess?: boolean;
  }
): Promise<{
  task_id: string;
  extraction_job_id: string;
  status: string;
  message: string;
}> {
  return fetchAPI<{
    task_id: string;
    extraction_job_id: string;
    status: string;
    message: string;
  }>(`/extractions/${extractionId}/analyze-nlp`, {
    method: 'POST',
    body: JSON.stringify(options || {}),
  });
}

/**
 * Cancel an in-progress NLP analysis job.
 * Preserves progress already made - features that have been analyzed keep their results.
 *
 * @param extractionId - The extraction job ID
 * @returns Control response with cancellation status
 */
export async function cancelNlpAnalysis(
  extractionId: string
): Promise<{
  extraction_job_id: string;
  action: string;
  previous_status: string | null;
  previous_progress: number | null;
  features_affected: number | null;
  message: string;
}> {
  return fetchAPI(`/extractions/${extractionId}/cancel-nlp`, {
    method: 'POST',
  });
}

/**
 * Reset NLP analysis status for an extraction job.
 * Allows resuming (skips already processed features) or restarting from scratch.
 *
 * @param extractionId - The extraction job ID
 * @param options - Reset options:
 *   - clear_feature_analysis: If true, clears NLP analysis from all features (restart from scratch)
 *                             If false (default), preserves existing analysis (resume where left off)
 * @returns Control response with reset status
 */
export async function resetNlpAnalysis(
  extractionId: string,
  options?: {
    clear_feature_analysis?: boolean;
  }
): Promise<{
  extraction_job_id: string;
  action: string;
  previous_status: string | null;
  previous_progress: number | null;
  features_affected: number | null;
  message: string;
}> {
  return fetchAPI(`/extractions/${extractionId}/reset-nlp`, {
    method: 'POST',
    body: JSON.stringify(options || {}),
  });
}
