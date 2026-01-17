/**
 * API client for SAE operations.
 *
 * This module provides functions to interact with the backend SAEs API.
 * All functions make REAL HTTP requests to the production backend.
 */

import {
  SAE,
  SAEListResponse,
  HFRepoPreviewRequest,
  HFRepoPreviewResponse,
  SAEDownloadRequest,
  SAEUploadRequest,
  SAEUploadResponse,
  SAEImportFromTrainingRequest,
  SAEImportFromFileRequest,
  SAEFeatureBrowserResponse,
  SAEDeleteRequest,
  SAEDeleteResponse,
  SAESource,
  SAEStatus,
} from '../types/sae';
import {
  ExtractionStatusResponse,
  BatchExtractionRequest,
  BatchExtractionResponse,
} from '../types/features';
import { fetchAPI, buildQueryString } from './client';

/**
 * Get list of SAEs with optional filters
 */
export async function getSAEs(params?: {
  skip?: number;
  limit?: number;
  search?: string;
  source?: SAESource;
  status?: SAEStatus;
  model_name?: string;
  sort_by?: string;
  order?: 'asc' | 'desc';
}): Promise<SAEListResponse> {
  const query = params ? buildQueryString(params) : '';
  const endpoint = `/saes${query ? `?${query}` : ''}`;
  return fetchAPI<SAEListResponse>(endpoint);
}

/**
 * Get a single SAE by ID
 */
export async function getSAE(id: string): Promise<SAE> {
  return fetchAPI<SAE>(`/saes/${id}`);
}

/**
 * Preview a HuggingFace repository to discover SAEs
 */
export async function previewHFRepository(
  request: HFRepoPreviewRequest
): Promise<HFRepoPreviewResponse> {
  return fetchAPI<HFRepoPreviewResponse>('/saes/hf/preview', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Download an SAE from HuggingFace
 */
export async function downloadSAE(
  request: SAEDownloadRequest
): Promise<SAE> {
  return fetchAPI<SAE>('/saes/download', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Upload an SAE to HuggingFace
 */
export async function uploadSAE(
  request: SAEUploadRequest
): Promise<SAEUploadResponse> {
  return fetchAPI<SAEUploadResponse>('/saes/upload', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Import an SAE from a completed training job
 */
export async function importSAEFromTraining(
  request: SAEImportFromTrainingRequest
): Promise<SAE> {
  return fetchAPI<SAE>('/saes/import/training', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Import an SAE from a local file
 */
export async function importSAEFromFile(
  request: SAEImportFromFileRequest
): Promise<SAE> {
  return fetchAPI<SAE>('/saes/import/file', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Delete a single SAE
 */
export async function deleteSAE(
  id: string,
  deleteFiles: boolean = true
): Promise<{ message: string }> {
  const query = deleteFiles ? '' : '?delete_files=false';
  return fetchAPI<{ message: string }>(`/saes/${id}${query}`, {
    method: 'DELETE',
  });
}

/**
 * Delete multiple SAEs
 */
export async function deleteSAEsBatch(
  request: SAEDeleteRequest,
  deleteFiles: boolean = true
): Promise<SAEDeleteResponse> {
  const query = deleteFiles ? '' : '?delete_files=false';
  return fetchAPI<SAEDeleteResponse>(`/saes/delete${query}`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Browse features in an SAE for steering
 */
export async function browseSAEFeatures(
  saeId: string,
  params?: {
    skip?: number;
    limit?: number;
    search?: string;
  }
): Promise<SAEFeatureBrowserResponse> {
  const query = params ? buildQueryString(params) : '';
  const endpoint = `/saes/${saeId}/features${query ? `?${query}` : ''}`;
  return fetchAPI<SAEFeatureBrowserResponse>(endpoint);
}

/**
 * Get SAEs that are ready for use in steering
 */
export async function getReadySAEs(
  modelName?: string
): Promise<SAEListResponse> {
  const params: Record<string, any> = {
    status: SAEStatus.READY,
    limit: 100,
  };
  if (modelName) {
    params.model_name = modelName;
  }
  const query = buildQueryString(params);
  return fetchAPI<SAEListResponse>(`/saes?${query}`);
}

// ============================================================================
// SAE Feature Extraction
// ============================================================================

// ExtractionStatusResponse is imported from '../types/features' (canonical source)

/**
 * Extraction configuration for SAE feature extraction
 */
export interface SAEExtractionConfig {
  evaluation_samples?: number;
  top_k_examples?: number;
  batch_size?: number;
  num_workers?: number;
  db_commit_batch?: number;
  filter_special?: boolean;
  filter_single_char?: boolean;
  filter_punctuation?: boolean;
  filter_numbers?: boolean;
  filter_fragments?: boolean;
  filter_stop_words?: boolean;
  context_prefix_tokens?: number;
  context_suffix_tokens?: number;
  min_activation_frequency?: number;
  auto_nlp?: boolean;  // Automatically run NLP analysis after extraction (default: true)
}

/**
 * Start feature extraction from an SAE
 */
export async function startSAEExtraction(
  saeId: string,
  datasetId: string,
  config: SAEExtractionConfig
): Promise<ExtractionStatusResponse> {
  const query = `dataset_id=${encodeURIComponent(datasetId)}`;
  return fetchAPI<ExtractionStatusResponse>(`/saes/${saeId}/extract-features?${query}`, {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

/**
 * Get extraction status for an SAE
 */
export async function getSAEExtractionStatus(
  saeId: string
): Promise<ExtractionStatusResponse> {
  return fetchAPI<ExtractionStatusResponse>(`/saes/${saeId}/extraction-status`);
}

/**
 * Cancel extraction for an SAE
 */
export async function cancelSAEExtraction(
  saeId: string
): Promise<{ message: string }> {
  return fetchAPI<{ message: string }>(`/saes/${saeId}/cancel-extraction`, {
    method: 'POST',
  });
}

/**
 * Start batch feature extraction for multiple SAEs
 *
 * Creates extraction jobs for all specified SAEs using the same dataset
 * and configuration. Jobs are queued and processed sequentially.
 */
export async function startBatchSAEExtraction(
  request: BatchExtractionRequest
): Promise<BatchExtractionResponse> {
  return fetchAPI<BatchExtractionResponse>('/saes/batch-extract-features', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}
