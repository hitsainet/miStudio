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
  /**
   * The worker stopped reporting and is presumed gone.
   *
   * Emitted by GET /task-queue/active, which reconciles a running row against
   * the Celery heartbeat. It was missing here while the backend already sent
   * it, so the value arrived untyped and any `switch` over this enum fell
   * through to its default — which is how a dead job rendered as "Queued".
   */
  ORPHANED = 'orphaned',
}

/**
 * Task type
 */
export enum TaskType {
  DOWNLOAD = 'download',
  TRAINING = 'training',
  EXTRACTION = 'extraction',
  TOKENIZATION = 'tokenization',
  LABELING = 'labeling',
  NEURONPEDIA_PUSH = 'neuronpedia_push',
  // J-space work. Prefixed so Active Operations can tell a 45-minute lens fit
  // apart from a training run at a glance — and so the J-Lens panel can filter
  // to just its own jobs without matching on entity ids.
  JLENS_FIT = 'jlens_fit',
  JLENS_BAND_REPORT = 'jlens_band_report',
  JLENS_INTERVENTION = 'jlens_intervention',
  JLENS_READOUT = 'jlens_readout',
  JLENS_PROBE = 'jlens_probe',
  JLENS_ACQUIRE = 'jlens_acquire',
  JLENS_PUBLISH = 'jlens_publish',
}

/**
 * Entity type
 */
export enum EntityType {
  MODEL = 'model',
  DATASET = 'dataset',
  TRAINING = 'training',
  EXTRACTION = 'extraction',
  LABELING = 'labeling',
  NEURONPEDIA = 'neuronpedia',
}

/**
 * Entity information associated with a task
 */
export interface EntityInfo {
  id?: string;
  name: string;
  repo_id?: string;
  hf_repo_id?: string;
  details?: string;
  status?: string;
  type?: string;

  /**
   * Live progress merged from the worker's own report, present only for a
   * RUNNING task that has reported at least once.
   *
   * Every field is optional and ABSENT MEANS UNKNOWN, never zero. A fit that
   * has not yet reported must not render as `0 / 1200`, which would claim it
   * had done nothing rather than that nothing is known yet.
   *
   * These ride on `entity_info` because it is free-form; `TaskQueueData` is a
   * closed Pydantic model and silently drops unknown top-level keys.
   */
  stage?: string;
  prompts_seen?: number;
  /** The DENOMINATOR. Without it a reader can show a percentage but not
   *  "634 / 1200" except by reconstructing it from a rounded number. */
  total_prompts?: number;
  last_delta?: number;
  /** The threshold `last_delta` is racing. A delta with no target cannot be
   *  judged by anyone reading it. */
  convergence_delta?: number;
  converged?: boolean;
  seconds_since_heartbeat?: number;
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
  /** False for rows federated from other job tables (trainings, extractions,
   *  labeling, pushes) — those are read-only in the task-queue API. */
  can_retry: boolean;
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
