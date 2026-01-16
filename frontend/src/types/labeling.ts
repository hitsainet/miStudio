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
  OPENAI_COMPATIBLE = 'openai_compatible',
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

  /** Labeling method (openai, openai_compatible, local, manual) */
  labeling_method: LabelingMethod;

  /** OpenAI model to use (default: gpt-4o-mini) */
  openai_model?: string;

  /** OpenAI API key (optional, uses server default if not provided) */
  openai_api_key?: string;

  /** OpenAI-compatible endpoint (e.g., /ollama/v1 for local proxy) */
  openai_compatible_endpoint?: string;

  /** OpenAI-compatible model (e.g., llama3.2) */
  openai_compatible_model?: string;

  /** Local model to use (default: meta-llama/Llama-3.2-1B) */
  local_model?: string;

  /** Prompt template ID to use for labeling (optional, uses default if not specified) */
  prompt_template_id?: string;

  /** Filter special tokens (<s>, </s>, etc.) from token analysis (default: true) */
  filter_special?: boolean;

  /** Filter single character tokens from token analysis (default: true) */
  filter_single_char?: boolean;

  /** Filter pure punctuation tokens from token analysis (default: true) */
  filter_punctuation?: boolean;

  /** Filter pure numeric tokens from token analysis (default: true) */
  filter_numbers?: boolean;

  /** Filter word fragments (BPE subwords) from token analysis (default: true) */
  filter_fragments?: boolean;

  /** Filter common stop words from token analysis (default: false) */
  filter_stop_words?: boolean;

  /** Save API requests to /tmp/ for testing and debugging (default: false) */
  save_requests_for_testing?: boolean;

  /** Sample rate for saving API requests (0.0-1.0, default: 1.0) */
  save_requests_sample_rate?: number;

  /** Export format for saved API requests: 'postman' (Postman collection), 'curl' (cURL command), or 'both' (default: both) */
  export_format?: 'postman' | 'curl' | 'both';

  /** Batch size for labeling (1-100, default: 10) */
  batch_size?: number;

  /** Number of activation examples per feature (10-50) */
  max_examples?: number;

  /** API request timeout in seconds (30-600, default: 120) */
  api_timeout?: number;

  /** Save poor quality labels for debugging (default: false) */
  save_poor_quality_labels?: boolean;

  /** Sample rate for saving poor quality labels (0.0-1.0, default: 1.0) */
  poor_quality_sample_rate?: number;
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

  /** OpenAI-compatible model used (if applicable) */
  openai_compatible_model?: string | null;

  /** Local model used (if applicable) */
  local_model?: string | null;

  /** Prompt template ID used (if applicable) */
  prompt_template_id?: string | null;

  // Token filtering configuration
  /** Filter special tokens */
  filter_special: boolean;

  /** Filter single character tokens */
  filter_single_char: boolean;

  /** Filter pure punctuation */
  filter_punctuation: boolean;

  /** Filter pure numeric tokens */
  filter_numbers: boolean;

  /** Filter word fragments */
  filter_fragments: boolean;

  /** Filter common stop words */
  filter_stop_words: boolean;

  /** Save API requests for testing */
  save_requests_for_testing: boolean;

  /** Export format for saved API requests */
  export_format: string;

  /** Save poor quality labels for debugging */
  save_poor_quality_labels: boolean;

  /** Sample rate for saving poor quality labels (0.0-1.0) */
  poor_quality_sample_rate: number;

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
