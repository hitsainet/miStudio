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

  // Handle 204 No Content responses (e.g., DELETE)
  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

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
  const queryParams = new URLSearchParams();

  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryParams.append(key, String(value));
      }
    });
  }

  const endpoint = `/models${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
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
 * Extract activations from a model
 */
export async function extractActivations(
  modelId: string,
  config: ActivationExtractionConfig
): Promise<ActivationExtractionResult> {
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
