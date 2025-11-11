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
