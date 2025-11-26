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

/**
 * Generate a steering comparison between baseline and steered outputs.
 */
export async function generateComparison(
  request: SteeringComparisonRequest
): Promise<SteeringComparisonResponse> {
  return fetchAPI<SteeringComparisonResponse>('/steering/compare', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Run a strength sweep for a single feature.
 */
export async function runStrengthSweep(
  request: SteeringStrengthSweepRequest
): Promise<StrengthSweepResponse> {
  return fetchAPI<StrengthSweepResponse>('/steering/sweep', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

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
