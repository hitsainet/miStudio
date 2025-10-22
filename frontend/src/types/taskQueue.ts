/**
 * Task Queue Types
 *
 * TypeScript interfaces for Task Queue feature.
 * Provides visibility and control over background operations.
 */

/**
 * Task queue entry status
 */
export enum TaskQueueStatus {
  QUEUED = 'queued',
  RUNNING = 'running',
  FAILED = 'failed',
  COMPLETED = 'completed',
  CANCELLED = 'cancelled',
}

/**
 * Task type
 */
export enum TaskType {
  DOWNLOAD = 'download',
  TRAINING = 'training',
  EXTRACTION = 'extraction',
  TOKENIZATION = 'tokenization',
}

/**
 * Entity type
 */
export enum EntityType {
  MODEL = 'model',
  DATASET = 'dataset',
  TRAINING = 'training',
  EXTRACTION = 'extraction',
}

/**
 * Entity information associated with a task
 */
export interface EntityInfo {
  id: string;
  name: string;
  repo_id?: string;
  hf_repo_id?: string;
  status: string;
  type: string;
}

/**
 * Task queue entry
 */
export interface TaskQueueEntry {
  id: string;
  task_id: string | null;
  task_type: TaskType;
  entity_id: string;
  entity_type: EntityType;
  status: TaskQueueStatus;
  progress: number | null;
  error_message: string | null;
  retry_params: Record<string, any> | null;
  retry_count: number;
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  updated_at: string | null;
  entity_info: EntityInfo | null;
}

/**
 * Task queue list response
 */
export interface TaskQueueListResponse {
  data: TaskQueueEntry[];
}

/**
 * Task queue single entry response
 */
export interface TaskQueueResponse {
  data: TaskQueueEntry;
}

/**
 * Retry request parameters
 */
export interface RetryRequest {
  param_overrides?: Record<string, any>;
}

/**
 * Retry response
 */
export interface RetryResponse {
  success: boolean;
  message: string;
  task_queue_id: string;
  celery_task_id: string;
  retry_count: number;
}
