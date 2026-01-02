/**
 * API client for Steering operations.
 *
 * This module provides functions to interact with the backend Steering API.
 * All functions make REAL HTTP requests to the production backend.
 */

import {
  SteeringComparisonRequest,
  SteeringComparisonResponse,
  SteeringStrengthSweepRequest,
  StrengthSweepResponse,
  SteeringExperiment,
  SteeringExperimentListResponse,
  SteeringExperimentSaveRequest,
} from '../types/steering';
import { fetchAPI, buildQueryString } from './client';

// ============================================================================
// Async Task Types (for Celery-based steering)
// ============================================================================

/**
 * Response from submitting an async steering task.
 */
export interface SteeringTaskResponse {
  task_id: string;
  task_type: 'compare' | 'sweep';
  status: string;
  websocket_channel: string;
  message: string;
  submitted_at: string;
}

/**
 * Status of an async steering task.
 */
export interface SteeringTaskStatus {
  task_id: string;
  status: 'pending' | 'started' | 'progress' | 'success' | 'failure' | 'revoked';
  percent: number;
  message: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
}

/**
 * Result of an async steering task.
 */
export interface SteeringResultResponse {
  task_id: string;
  status: SteeringTaskStatus;
  result?: SteeringComparisonResponse | StrengthSweepResponse;
}

/**
 * Response from cancelling a steering task.
 */
export interface SteeringCancelResponse {
  task_id: string;
  status: string;
  message: string;
}

// NOTE: The sync generateComparison() function has been removed.
// Use submitAsyncComparison() instead, which uses the Celery-based async API.
// This prevents zombie GPU processes from blocking model.generate() calls.

// NOTE: The sync runStrengthSweep() function has been removed.
// Use submitAsyncSweep() instead, which uses the Celery-based async API.

/**
 * Get list of saved steering experiments with optional filters.
 */
export async function getExperiments(params?: {
  skip?: number;
  limit?: number;
  search?: string;
  sae_id?: string;
  model_id?: string;
  tag?: string;
}): Promise<SteeringExperimentListResponse> {
  const query = params ? buildQueryString(params) : '';
  const endpoint = `/steering/experiments${query ? `?${query}` : ''}`;
  return fetchAPI<SteeringExperimentListResponse>(endpoint);
}

/**
 * Get a single steering experiment by ID.
 */
export async function getExperiment(id: string): Promise<SteeringExperiment> {
  return fetchAPI<SteeringExperiment>(`/steering/experiments/${id}`);
}

/**
 * Save a steering experiment.
 */
export async function saveExperiment(
  request: SteeringExperimentSaveRequest
): Promise<SteeringExperiment> {
  return fetchAPI<SteeringExperiment>('/steering/experiments', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Delete a steering experiment.
 */
export async function deleteExperiment(id: string): Promise<{ message: string }> {
  return fetchAPI<{ message: string }>(`/steering/experiments/${id}`, {
    method: 'DELETE',
  });
}

/**
 * Delete multiple steering experiments.
 */
export async function deleteExperimentsBatch(
  ids: string[]
): Promise<{ deleted_count: number; message: string }> {
  return fetchAPI<{ deleted_count: number; message: string }>('/steering/experiments/delete', {
    method: 'POST',
    body: JSON.stringify({ ids }),
  });
}

/**
 * Abort an in-progress steering comparison.
 */
export async function abortComparison(comparisonId: string): Promise<{ message: string }> {
  return fetchAPI<{ message: string }>(`/steering/compare/${comparisonId}/abort`, {
    method: 'POST',
  });
}

/**
 * Get the status of an in-progress comparison.
 */
export async function getComparisonStatus(
  comparisonId: string
): Promise<{ status: string; progress: number; message: string | null }> {
  return fetchAPI<{ status: string; progress: number; message: string | null }>(
    `/steering/compare/${comparisonId}/status`
  );
}

// ============================================================================
// Async Steering API (Celery-based with WebSocket progress)
// ============================================================================

/**
 * Submit an async steering comparison task.
 * Returns immediately with task_id. Use WebSocket or polling for results.
 */
export async function submitAsyncComparison(
  request: SteeringComparisonRequest
): Promise<SteeringTaskResponse> {
  return fetchAPI<SteeringTaskResponse>('/steering/async/compare', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Submit an async strength sweep task.
 * Returns immediately with task_id. Use WebSocket or polling for results.
 */
export async function submitAsyncSweep(
  request: SteeringStrengthSweepRequest
): Promise<SteeringTaskResponse> {
  return fetchAPI<SteeringTaskResponse>('/steering/async/sweep', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Get the result of an async steering task.
 * Returns status and result (if completed).
 */
export async function getTaskResult(taskId: string): Promise<SteeringResultResponse> {
  return fetchAPI<SteeringResultResponse>(`/steering/async/result/${taskId}`);
}

/**
 * Cancel a pending or running steering task.
 */
export async function cancelTask(taskId: string): Promise<SteeringCancelResponse> {
  return fetchAPI<SteeringCancelResponse>(`/steering/async/task/${taskId}`, {
    method: 'DELETE',
  });
}
