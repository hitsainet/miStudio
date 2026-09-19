/**
 * API client for dataset operations.
 *
 * This module provides functions to interact with the backend datasets API.
 */

import {
  Dataset,
  DatasetDownloadRequest,
  DatasetListResponse,
  DatasetSample,
} from '../types/dataset';
import { fetchAPI, buildQueryString } from './client';

/**
 * Get list of datasets with optional filters
 */
export async function getDatasets(params?: {
  skip?: number;
  limit?: number;
  search?: string;
  source?: string;
  status?: string;
  sort_by?: string;
  order?: 'asc' | 'desc';
}): Promise<DatasetListResponse> {
  const query = params ? buildQueryString(params) : '';
  const endpoint = `/datasets${query ? `?${query}` : ''}`;
  return fetchAPI<DatasetListResponse>(endpoint);
}

/**
 * Get a single dataset by ID
 */
export async function getDataset(id: string): Promise<Dataset> {
  return fetchAPI<Dataset>(`/datasets/${id}`);
}

/**
 * Download a dataset from HuggingFace
 */
export async function downloadDataset(
  request: DatasetDownloadRequest
): Promise<Dataset> {
  return fetchAPI<Dataset>('/datasets/download', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Delete a dataset
 */
export async function deleteDataset(id: string): Promise<void> {
  return fetchAPI<void>(`/datasets/${id}`, {
    method: 'DELETE',
  });
}

/**
 * Get dataset samples with pagination
 */
export async function getDatasetSamples(
  id: string,
  params?: {
    skip?: number;
    limit?: number;
    search?: string;
  }
): Promise<{ data: DatasetSample[]; total: number }> {
  const query = params ? buildQueryString(params) : '';
  const endpoint = `/datasets/${id}/samples${query ? `?${query}` : ''}`;
  return fetchAPI<{ data: DatasetSample[]; total: number }>(endpoint);
}

/**
 * Get dataset statistics
 */
export async function getDatasetStatistics(id: string): Promise<any> {
  return fetchAPI<any>(`/datasets/${id}/statistics`);
}

/**
 * Trigger dataset tokenization
 */
export async function tokenizeDataset(
  id: string,
  settings: {
    max_length?: number;
    truncation?: boolean;
    padding?: boolean;
    add_special_tokens?: boolean;
  }
): Promise<Dataset> {
  return fetchAPI<Dataset>(`/datasets/${id}/tokenize`, {
    method: 'POST',
    body: JSON.stringify(settings),
  });
}

/**
 * Cancel an in-progress dataset download or tokenization.
 *
 * This will stop the background task, clean up partial files,
 * and update the dataset status to ERROR with "Cancelled by user".
 *
 * @param id - Dataset ID
 * @returns Cancellation status
 */
export async function cancelDatasetDownload(id: string): Promise<{
  dataset_id: string;
  status: string;
  message: string;
}> {
  return fetchAPI<{
    dataset_id: string;
    status: string;
    message: string;
  }>(`/datasets/${id}/cancel`, {
    method: 'DELETE',
  });
}
