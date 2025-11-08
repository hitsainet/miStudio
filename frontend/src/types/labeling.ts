/**
 * Labeling Types
 *
 * TypeScript interfaces for independent semantic labeling of SAE features.
 * Matches backend Pydantic schemas in src/schemas/labeling.py and src/models/labeling_job.py
 *
 * Backend API Contract:
 * - POST /api/v1/labeling - Start labeling job for an extraction
 * - GET /api/v1/labeling/:id - Get labeling job status
 * - GET /api/v1/labeling - List all labeling jobs
 * - POST /api/v1/labeling/:id/cancel - Cancel labeling job
 * - DELETE /api/v1/labeling/:id - Delete labeling job (keeps labels intact)
 * - POST /api/v1/extractions/:id/label - Convenience endpoint to label extraction
 *
 * WebSocket Events:
 * - Channel: labeling/{labeling_job_id}/progress
 *   - labeling:started - Job started
 *   - labeling:progress - Progress update
 *   - labeling:completed - Labeling completed
 *   - labeling:failed - Labeling failed
 *
 * Status transitions: queued -> labeling -> completed/failed/cancelled
 */

/**
 * Labeling job status.
 * Matches backend LabelingStatus enum.
 */
export enum LabelingStatus {
  QUEUED = 'queued',
  LABELING = 'labeling',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

/**
 * Labeling method.
 * Matches backend LabelingMethod enum.
 */
export enum LabelingMethod {
  OPENAI = 'openai',
  LOCAL = 'local',
  MANUAL = 'manual',
}

/**
 * Labeling configuration request.
 * Matches backend LabelingConfigRequest schema.
 */
export interface LabelingConfigRequest {
  /** Extraction job ID to label features from */
  extraction_job_id: string;

  /** Labeling method (openai, local, manual) */
  labeling_method: LabelingMethod;

  /** OpenAI model to use (default: gpt-4o-mini) */
  openai_model?: string;

  /** OpenAI API key (optional, uses server default if not provided) */
  openai_api_key?: string;

  /** Local model to use (default: meta-llama/Llama-3.2-1B) */
  local_model?: string;

  /** Batch size for labeling (1-100, default: 10) */
  batch_size?: number;
}

/**
 * Labeling job.
 * Matches backend LabelingJob model and LabelingStatusResponse schema.
 */
export interface LabelingJob {
  /** Labeling job ID (format: label_{extraction_id}_{timestamp}) */
  id: string;

  /** Extraction job ID being labeled */
  extraction_job_id: string;

  // Configuration
  /** Labeling method used */
  labeling_method: LabelingMethod;

  /** OpenAI model used (if applicable) */
  openai_model?: string | null;

  /** Local model used (if applicable) */
  local_model?: string | null;

  // Status and progress
  /** Current labeling status */
  status: LabelingStatus;

  /** Labeling progress (0-1) */
  progress: number;

  /** Number of features labeled so far */
  features_labeled: number;

  /** Total features to label */
  total_features?: number | null;

  // Results
  /** Error message if failed */
  error_message?: string | null;

  /** Labeling statistics */
  statistics?: LabelingStatistics | null;

  // Celery
  /** Celery task ID */
  celery_task_id?: string | null;

  // Timestamps
  /** Job creation timestamp */
  created_at: string;

  /** Last update timestamp */
  updated_at: string;

  /** Labeling completion timestamp */
  completed_at?: string | null;
}

/**
 * Labeling job statistics.
 * Included in LabelingJob.statistics when complete.
 */
export interface LabelingStatistics {
  /** Total features processed */
  total_features: number;

  /** Successfully labeled features */
  successfully_labeled: number;

  /** Failed label generations */
  failed_labels: number;

  /** Average label length in characters */
  avg_label_length: number;

  /** Labeling duration in seconds */
  labeling_duration_seconds: number;

  /** Labeling method used */
  labeling_method: string;
}

/**
 * Paginated labeling list response.
 * Matches backend LabelingListResponse schema.
 */
export interface LabelingListResponse {
  /** List of labeling jobs */
  data: LabelingJob[];

  /** Pagination metadata */
  meta: {
    total: number;
    limit: number;
    offset: number;
  };
}

/**
 * Labeling list query parameters.
 */
export interface LabelingListParams {
  /** Filter by extraction job ID */
  extraction_job_id?: string;

  /** Maximum number of results (1-100) */
  limit?: number;

  /** Number of results to skip */
  offset?: number;
}

/**
 * WebSocket labeling progress event payload.
 */
export interface LabelingProgressEvent {
  labeling_job_id: string;
  features_labeled: number;
  total_features: number;
  progress: number;
  status: LabelingStatus;
}

/**
 * WebSocket labeling completed event payload.
 */
export interface LabelingCompletedEvent {
  labeling_job_id: string;
  statistics: LabelingStatistics;
}

/**
 * WebSocket labeling failed event payload.
 */
export interface LabelingFailedEvent {
  labeling_job_id: string;
  error_message: string;
}
