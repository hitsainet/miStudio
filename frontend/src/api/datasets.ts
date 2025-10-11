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
import { API_BASE_URL } from '../config/api';

// API endpoints are prefixed with /api/v1
const API_V1_BASE = `${API_BASE_URL}/api/v1`;

/**
 * Fetch helper with authentication and error handling
 */
async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = localStorage.getItem('auth_token');

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_V1_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      detail: `HTTP error! status: ${response.status}`,
    }));
    throw new Error(error.detail || error.message || 'API request failed');
  }

  return response.json();
}

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
  const queryParams = new URLSearchParams();

  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryParams.append(key, String(value));
      }
    });
  }

  const endpoint = `/datasets${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
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
  const queryParams = new URLSearchParams();

  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryParams.append(key, String(value));
      }
    });
  }

  const endpoint = `/datasets/${id}/samples${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
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
