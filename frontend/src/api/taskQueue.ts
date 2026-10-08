/**
 * Task Queue API client.
 *
 * This module provides functions to interact with the task queue endpoints
 * for viewing and managing background operations.
 */

import axios from 'axios';
import { API_BASE_URL } from '../config/api';
import {
  TaskQueueListResponse,
  TaskQueueResponse,
  RetryRequest,
  RetryResponse,
} from '../types/taskQueue';

const API_URL = `${API_BASE_URL}/api/v1/task-queue`;

/**
 * Get all task queue entries with optional filtering.
 *
 * @param status - Optional status filter (queued, running, failed, completed, cancelled)
 * @param entityType - Optional entity type filter (model, dataset, training, extraction)
 * @returns List of task queue entries
 */
export async function getTaskQueue(
  status?: string,
  entityType?: string
): Promise<TaskQueueListResponse> {
  const params = new URLSearchParams();
  if (status) params.append('status', status);
  if (entityType) params.append('entity_type', entityType);

  const url = params.toString() ? `${API_URL}?${params.toString()}` : API_URL;
  const response = await axios.get<TaskQueueListResponse>(url);
  return response.data;
}

/**
 * Get all failed task queue entries.
 *
 * @returns List of failed tasks
 */
export async function getFailedTasks(): Promise<TaskQueueListResponse> {
  const response = await axios.get<TaskQueueListResponse>(`${API_URL}/failed`);
  return response.data;
}

/**
 * Get all active (queued or running) task queue entries.
 *
 * @returns List of active tasks
 */
export async function getActiveTasks(): Promise<TaskQueueListResponse> {
  const response = await axios.get<TaskQueueListResponse>(`${API_URL}/active`);
  return response.data;
}

/**
 * Get a specific task queue entry by ID.
 *
 * @param taskQueueId - Task queue entry ID
 * @returns Task queue entry
 */
export async function getTaskById(taskQueueId: string): Promise<TaskQueueResponse> {
  const response = await axios.get<TaskQueueResponse>(`${API_URL}/${taskQueueId}`);
  return response.data;
}

/**
 * Retry a failed task.
 *
 * @param taskQueueId - Task queue entry ID
 * @param request - Optional parameter overrides for retry
 * @returns Retry response
 */
export async function retryTask(
  taskQueueId: string,
  request?: RetryRequest
): Promise<RetryResponse> {
  const response = await axios.post<RetryResponse>(
    `${API_URL}/${taskQueueId}/retry`,
    request || {}
  );
  return response.data;
}

/**
 * Delete a task queue entry.
 *
 * @param taskQueueId - Task queue entry ID
 */
export async function deleteTask(taskQueueId: string): Promise<void> {
  await axios.delete(`${API_URL}/${taskQueueId}`);
}

/**
 * Clear one federated failure from the Monitor.
 *
 * Federated rows (trainings, extractions, labeling, Neuronpedia pushes) live in
 * other tables and have no DELETE here, so this records a dismissal marker. The
 * job row keeps its status and error message; only the listing hides it.
 */
export async function dismissFailedTask(
  taskType: string,
  sourceId: string,
): Promise<void> {
  await axios.post(
    `${API_URL}/failed/${encodeURIComponent(taskType)}/${encodeURIComponent(sourceId)}/dismiss`,
  );
}

/** Clear every federated failure currently listed. Returns how many. */
export async function dismissAllFailedTasks(): Promise<number> {
  const response = await axios.post<{ dismissed_count: number }>(
    `${API_URL}/failed/dismiss-all`,
  );
  return response.data.dismissed_count;
}
