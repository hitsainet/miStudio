/**
 * Training Types
 *
 * TypeScript interfaces for SAE Training feature.
 * Matches backend Pydantic schemas in src/schemas/training.py and src/models/training.py
 *
 * Backend API Contract:
 * - GET /api/v1/trainings - List all training jobs
 * - GET /api/v1/trainings/:id - Get training job details
 * - POST /api/v1/trainings - Create and start new training job
 * - PATCH /api/v1/trainings/:id - Update training job
 * - DELETE /api/v1/trainings/:id - Delete training job
 * - POST /api/v1/trainings/:id/control - Control training (pause/resume/stop)
 * - GET /api/v1/trainings/:id/metrics - Get training metrics
 * - GET /api/v1/trainings/:id/checkpoints - List checkpoints
 * - GET /api/v1/trainings/:id/checkpoints/best - Get best checkpoint
 *
 * WebSocket Events:
 * - Channel: trainings/{training_id}/progress
 *   - training:created - Job created
 *   - training:progress - Progress update (every 100 steps)
 *   - training:status_changed - Status changed (pause/resume/stop)
 *   - training:completed - Training completed
 *   - training:failed - Training failed
 * - Channel: trainings/{training_id}/checkpoints
 *   - checkpoint:created - Checkpoint saved
 *
 * Status transitions: pending -> initializing -> running -> completed/failed/cancelled
 * Can pause from running -> paused, resume from paused -> running
 */

/**
 * Training job status.
 * Matches backend TrainingStatus enum.
 */
export enum TrainingStatus {
  PENDING = 'pending',
  INITIALIZING = 'initializing',
  RUNNING = 'running',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

/**
 * SAE training framework types.
 * Each framework has its own optimizer settings, loss formulation,
 * and tailored hyperparameters based on published papers.
 *
 * Matches backend SAEArchitectureType enum.
 */
export enum SAEArchitectureType {
  STANDARD_SAELENS = 'standard_saelens',    // Bricken et al. 2023 — L1 penalty, constant_norm_rescale
  STANDARD_ANTHROPIC = 'standard_anthropic', // Templeton et al. 2024 — L1 penalty, anthropic_rescale, L1~5.0
  JUMPRELU = 'jumprelu',                     // Rajamanoharan et al. 2024 — L0 via STE, learnable thresholds
  TOPK = 'topk',                             // Gao et al. 2024 — structural sparsity, no penalty, aux loss
  SKIP = 'skip',                             // Community variant — L1 + residual skip connection
  TRANSCODER = 'transcoder',                 // Dunefsky et al. 2024 — predicts MLP output from MLP input
  STANDARD = 'standard',                     // Backward compat alias → maps to standard_saelens
}

/**
 * Training hyperparameters configuration.
 * Matches backend TrainingHyperparameters schema.
 */
/**
 * Activation hook types for SAE training.
 * Determines which activation sites to capture during training.
 */
export type HookType = 'residual' | 'mlp' | 'attention';

export interface HyperparametersConfig {
  // SAE Architecture
  /** Hidden dimension (input/output size) */
  hidden_dim: number;
  /** Latent dimension (SAE width, typically 8-32x hidden_dim) */
  latent_dim: number;
  /** SAE architecture type */
  architecture_type: SAEArchitectureType;

  // Layer configuration
  /** List of layer indices to train SAEs on (e.g., [0, 6, 12]) */
  training_layers: number[];
  /** Hook types to train SAEs on (default: ['residual']). Multiple hook types create separate SAEs per layer/hook combination. */
  hook_types?: HookType[];

  // Sparsity (L1-based frameworks: standard_saelens, standard_anthropic, skip, transcoder)
  /** L1 sparsity penalty coefficient. Optional — not used by TopK. */
  l1_alpha?: number;
  /** Target L0 sparsity (fraction of active features, 0-1) */
  target_l0?: number;
  /** @deprecated Use top_k (integer count) instead. Percentage-based TopK sparsity. */
  top_k_sparsity?: number;

  // TopK-specific parameters (Gao et al. 2024)
  /** Number of active features per sample for TopK architecture */
  top_k?: number;
  /** Number of dead features for auxiliary loss (default: top_k) */
  aux_k?: number;
  /** Auxiliary dead feature loss coefficient (default: 1/32) */
  aux_loss_alpha?: number;
  /** Adam optimizer epsilon (TopK uses 6.25e-10) */
  adam_epsilon?: number;

  // JumpReLU-specific parameters (Rajamanoharan et al. 2024)
  /** Initial threshold value for JumpReLU activation (default: 0.5) */
  initial_threshold?: number;
  /** KDE bandwidth for STE gradient estimation in JumpReLU (default: 0.01) */
  bandwidth?: number;
  /** L0 sparsity coefficient for JumpReLU (default: 1e-4). */
  sparsity_coeff?: number;
  /** Whether to normalize decoder columns to unit norm */
  normalize_decoder?: boolean;
  /** Activation normalization method (constant_norm_rescale, anthropic_rescale, none) */
  normalize_activations?: string;

  // Training
  /** Initial learning rate */
  learning_rate: number;
  /** Training batch size */
  batch_size: number;
  /** Total training steps */
  total_steps: number;
  /** Linear warmup steps for learning rate */
  warmup_steps?: number;
  /** Sparsity warmup steps: linearly ramp L1/L0 penalty from 0 to full value. Prevents dead neurons. */
  sparsity_warmup_steps?: number;

  // Optimization
  /** Weight decay (L2 regularization) */
  weight_decay?: number;
  /** Gradient clipping norm */
  grad_clip_norm?: number;

  // Checkpointing
  /** Save checkpoint every N steps */
  checkpoint_interval?: number;
  /** Log metrics every N steps */
  log_interval?: number;

  // Dead neuron handling
  /** Steps before neuron considered dead */
  dead_neuron_threshold?: number;
  /** Resample dead neurons during training */
  resample_dead_neurons?: boolean;
  /** Resample dead neurons every N steps */
  resample_interval?: number;
}

/**
 * Training job.
 * Matches backend Training model and TrainingResponse schema.
 */
export interface Training {
  /** Training job ID (format: train_{uuid}) */
  id: string;
  /** Model ID being trained on */
  model_id: string;
  /** Dataset IDs for training data (supports multiple datasets) */
  dataset_ids: string[];
  /** Primary dataset ID (backward compat, first in dataset_ids) */
  dataset_id: string;
  /** Activation extraction ID (backward compat, first in extraction_ids) */
  extraction_id?: string | null;
  /** Extraction IDs for cached activations (one per dataset) */
  extraction_ids?: string[] | null;

  // Status and progress
  /** Current training status */
  status: TrainingStatus;
  /** Training progress (0-100) */
  progress: number;
  /** Current training step */
  current_step: number;
  /** Total planned training steps */
  total_steps: number;
  /**
   * Checkpoint step this run was finalized from after being stopped early.
   * null/undefined when the run completed normally. status is 'completed' in
   * both cases, so this is what distinguishes a full run from a salvaged one.
   */
  finalized_from_step?: number | null;

  // Hyperparameters
  /** Training hyperparameters */
  hyperparameters: HyperparametersConfig;

  // Current metrics (latest values)
  /** Current total loss */
  current_loss?: number | null;
  /** Current reconstruction loss component */
  current_reconstruction_loss?: number | null;
  /** Current L1 loss component (l1_alpha * l1_penalty) */
  current_l1_loss?: number | null;
  /** Current L1 alpha (sparsity coefficient, may be warming up) */
  current_l1_alpha?: number | null;
  /** Current L0 sparsity (fraction of active features) */
  current_l0_sparsity?: number | null;
  current_fvu?: number | null;
  /** Current dead neuron count */
  current_dead_neurons?: number | null;
  /** Current learning rate */
  current_learning_rate?: number | null;

  // Error handling
  /** Error message if failed */
  error_message?: string | null;

  // Paths
  /** Checkpoint directory path */
  checkpoint_dir?: string | null;
  /** Logs file path */
  logs_path?: string | null;

  // Celery
  /** Celery task ID */
  celery_task_id?: string | null;

  // Timestamps
  /** Job creation timestamp */
  created_at: string;
  /** Last update timestamp */
  updated_at: string;
  /** Training start timestamp */
  started_at?: string | null;
  /** Training completion timestamp */
  completed_at?: string | null;
}

/**
 * Training metric record (time-series data point).
 * Matches backend TrainingMetric model and TrainingMetricResponse schema.
 */
export interface TrainingMetric {
  /** Metric record ID */
  id: number;
  /** Training job ID */
  training_id: string;
  /** Training step */
  step: number;
  /** Metric collection timestamp */
  timestamp: string;

  // Loss metrics
  /** Total reconstruction loss */
  loss: number;
  /** Reconstruction component of loss */
  loss_reconstructed?: number | null;
  /** Zero ablation loss */
  loss_zero?: number | null;

  // Sparsity metrics
  /** L0 sparsity (fraction of active features) */
  l0_sparsity?: number | null;
  /** L1 sparsity penalty */
  l1_sparsity?: number | null;
  /** Dead neuron count */
  dead_neurons?: number | null;

  // Training dynamics
  /** Learning rate */
  learning_rate?: number | null;
  /** Gradient norm */
  grad_norm?: number | null;

  // Resource metrics
  /** GPU memory usage in MB */
  gpu_memory_used_mb?: number | null;
  /** Training throughput (samples/sec) */
  samples_per_second?: number | null;
}

/**
 * Training checkpoint.
 * Matches backend Checkpoint model and CheckpointResponse schema.
 */
export interface Checkpoint {
  /** Checkpoint ID (format: ckpt_{uuid}) */
  id: string;
  /** Training job ID */
  training_id: string;
  /** Training step at checkpoint */
  step: number;

  // Metrics at checkpoint
  /** Loss at checkpoint */
  loss: number;
  /** L0 sparsity at checkpoint */
  l0_sparsity?: number | null;

  // File storage
  /** Path to .safetensors file */
  storage_path: string;
  /** Checkpoint file size in bytes */
  file_size_bytes?: number | null;

  // Checkpoint metadata
  /** Whether this is the best checkpoint */
  is_best: boolean;
  /** Additional checkpoint metadata */
  extra_metadata?: Record<string, any> | null;

  // Timestamp
  /** Checkpoint creation timestamp */
  created_at: string;
}

/**
 * Training creation request.
 * Matches backend TrainingCreate schema.
 */
export interface TrainingCreateRequest {
  /** Model ID to train SAE on */
  model_id: string;
  /** Dataset IDs for training data (supports multiple datasets) */
  dataset_ids: string[];
  /** Extraction IDs for cached activations (one per dataset) */
  extraction_ids?: string[];
  /** Training hyperparameters */
  hyperparameters: HyperparametersConfig;
}

/**
 * Training control request.
 * Matches backend TrainingControlRequest schema.
 */
export interface TrainingControlRequest {
  /**
   * Control action to perform.
   * 'stop' cancels the run; 'stop_and_finalize' also writes community_format
   * from the newest checkpoint so the SAE stays importable.
   */
  action: 'pause' | 'resume' | 'stop' | 'stop_and_finalize';
}

/**
 * Training control response.
 * Matches backend TrainingControlResponse schema.
 */
export interface TrainingControlResponse {
  /** Whether the action succeeded */
  success: boolean;
  /** Training job ID */
  training_id: string;
  /** Action that was performed */
  action: string;
  /** New training status */
  status: TrainingStatus;
  /** Additional message */
  message?: string;
}

/**
 * Paginated training list response.
 * Matches backend TrainingListResponse schema.
 */
export interface TrainingListResponse {
  /** List of training jobs */
  data: Training[];
  /** Pagination metadata */
  pagination: {
    total: number;
    page: number;
    limit: number;
    total_pages: number;
  };
  /** Status counts for filtering */
  status_counts: {
    all: number;
    running: number;
    completed: number;
    failed: number;
  };
}

/**
 * Training metrics list response.
 * Matches backend TrainingMetricsListResponse schema.
 */
export interface TrainingMetricsListResponse {
  /** List of training metrics */
  data: TrainingMetric[];
  /** Pagination metadata (optional) */
  pagination?: {
    total: number;
    page: number;
    limit: number;
    total_pages: number;
  };
}

/**
 * Checkpoint list response.
 * Matches backend CheckpointListResponse schema.
 */
export interface CheckpointListResponse {
  /** List of checkpoints */
  data: Checkpoint[];
  /** Pagination metadata (optional) */
  pagination?: {
    total: number;
    page: number;
    limit: number;
    total_pages: number;
  };
}

/**
 * WebSocket training progress event payload.
 */
export interface TrainingProgressEvent {
  training_id: string;
  current_step: number;
  total_steps: number;
  progress: number;
  loss: number;
  reconstruction_loss?: number | null;
  l1_loss?: number | null;
  l1_alpha?: number | null;
  l0_sparsity: number;
  /** Fraction of Variance Unexplained: 0 = perfect reconstruction, 1 = mean-only.
   *  Optional because only architectures that compute it (JumpReLU) report one. */
  fvu?: number | null;
  dead_neurons: number;
  learning_rate: number;
}

/**
 * WebSocket checkpoint created event payload.
 */
export interface CheckpointCreatedEvent {
  training_id: string;
  checkpoint_id: string;
  step: number;
  loss: number;
  is_best: boolean;
  storage_path: string;
}

/**
 * Result of GET /trainings/{id}/checkpoints/prune-preview.
 * Read-only report of what the retention policy WOULD delete.
 */
export interface CheckpointPrunePreview {
  training_id: string;
  policy: {
    enabled: boolean;
    dry_run: boolean;
    keep_last: number;
    keep_best: boolean;
    min_age_hours: number;
  };
  prunable_steps: number[];
  kept_steps: number[];
  checkpoint_count: number;
  estimated_bytes: number;
  skipped_reason: string | null;
}
