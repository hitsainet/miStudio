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

  /** Maximum tokens in LLM response (50-8000, default: 300) */
  max_tokens?: number;

  /** Batch size for labeling (1-100, default: 10) */
  /**
   * Skip features that already carry a verdict.
   *
   * Defaults to false — a full relabel, which is what this endpoint has
   * always done. True fills only the gaps.
   */
  skip_adjudicated?: boolean;

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

  /**
   * OpenAI-compatible endpoint used (if applicable).
   *
   * Part of WHICH JUDGE ran. A resume that cannot reproduce it is not a
   * resume — it is a second, different experiment on the remainder. The
   * encrypted `openai_api_key` is deliberately never returned; the worker
   * resolves it from the job, app settings or the environment.
   */
  openai_compatible_endpoint?: string | null;

  /** API request timeout in seconds, also part of the judge's configuration. */
  api_timeout?: number | null;

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

  /** Maximum tokens in LLM response */
  max_tokens: number;

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

  // Extraction context (populated by API enrichment)
  /** Model name (from training or external SAE) */
  model_name?: string | null;

  /** Layer index from extraction job */
  layer_index?: number | null;

  /** Hook type from extraction job (e.g., 'residual', 'mlp', 'attention') */
  hook_type?: string | null;

  /** External SAE name (if applicable) */
  sae_name?: string | null;
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

/**
 * What still needs labeling in one extraction, and the batch a resume would take.
 *
 * Returned by `GET /labeling/{extraction_job_id}/coverage`. This is the answer
 * to "what is left?", a question that could not be asked before: the outcome of
 * a labeling attempt was not recorded anywhere, and a failure was written as a
 * fake label with `label_source` and `labeled_at` both set — so every existing
 * count read 16,824 failed features across the estate as finished work.
 */
/**
 * One normalised failure reason and what it cost.
 *
 * `reason` is the first colon-delimited segment of the stored error — the
 * exception class, or the whole fixed sentence — so a variable tail (host,
 * port, feature id) does not split one cause into thousands of singleton
 * groups. `(no reason recorded)` is the backfilled placeholder: on the L46
 * extraction that is 14,560 of them, and it is the most important row here.
 */
export interface LabelingFailureReason {
  reason: string;
  count: number;
  /** Real features carrying this reason, so a count can be checked, not believed. */
  sample_feature_ids: string[];
}

export interface LabelingCoverage {
  extraction_job_id: string;
  /** Every feature in the extraction. */
  total: number;
  /** Raw per-status counts: pending | in_progress | succeeded | failed | skipped. */
  by_status: Record<string, number>;
  /**
   * Settled: a verdict exists, or the feature was deliberately left alone.
   * Includes `uninterpretable` — the judge's honest "no coherent pattern",
   * which is a RESULT and must never be redone.
   */
  adjudicated: number;
  /** Still lacking a verdict, whether or not a resume will take it. */
  outstanding: number;
  /**
   * What a resume would ACTUALLY take — measured with the batch query's own
   * predicate, so the button and the job cannot disagree.
   */
  remaining: number;
  /**
   * Outstanding features that have used up their retries. Reported rather than
   * hidden: `outstanding` minus `exhausted` is `remaining`, and without this a
   * card showing "39 failed" and no Resume button is arithmetic an operator
   * cannot reconcile.
   */
  exhausted: number;
  /** Claimed by a running job right now. */
  in_progress: number;
  /** A status the backend has not been taught about. Reported, never absorbed. */
  unclassified: number;
  /**
   * Adjudicated by a DIFFERENT judge than the one asked about. `null` unless a
   * prompt fingerprint and model were supplied — reporting 0 without them would
   * read as "nothing is stale", which is not known.
   */
  stale: number | null;
  failures_without_a_recorded_reason: number;
  /** Set when some failures predate per-feature error capture. */
  caveat: string | null;
  /**
   * Why the failures failed, commonest first. Answers "will re-running these
   * work?" before spending 32 GPU-hours finding out empirically.
   */
  failure_reasons: LabelingFailureReason[];

  /** Ready to POST to `/labeling/panel` unchanged. */
  resume_feature_ids: string[];
}


/**
 * A multi-batch labeling resume.
 *
 * Finishing a 53k-feature extraction is ~27 batches of 2000 at ~8 s/feature —
 * about 59 GPU-hours. Each batch is its own Celery task, because a single task
 * looping over them would exceed the 10 h soft limit and strand its message for
 * the 12 h visibility timeout.
 */
export interface LabelingResumeSweep {
  id: string;
  extraction_job_id: string;
  /** running | completed | failed | cancelled */
  status: string;
  batches_done: number;
  max_batches: number;
  batch_size: number;
  /** Counted from what each batch WROTE, never from batch count. */
  features_labeled: number;
  features_failed: number;
  last_labeling_job_id: string | null;
  cancel_requested_at: string | null;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
}


/**
 * What a job's labeling endpoint can serve right now.
 *
 * Resume defaults to the judge a job used, which is right when comparing halves
 * of one run — and useless when that judge is gone, which is exactly when
 * someone needs to resume. This is what lets a resume name a different model.
 *
 * `features.label_model` and `label_prompt_fingerprint` are recorded per
 * feature, so an extraction labelled by two judges stays honest about which
 * produced what.
 */
export interface AvailableJudges {
  labeling_job_id: string;
  endpoint: string | null;
  /** What the job originally used. May no longer be served. */
  original_model: string | null;
  /** True when the original is still available — a plain resume works. */
  original_available: boolean;
  models: string[];
  reachable: boolean;
  detail: string | null;
}
