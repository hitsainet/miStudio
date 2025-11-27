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
 * SAE architecture types.
 * Matches backend SAEArchitectureType enum.
 */
export enum SAEArchitectureType {
  STANDARD = 'standard',
  SKIP = 'skip',
  TRANSCODER = 'transcoder',
  JUMPRELU = 'jumprelu',  // Gemma Scope architecture with learnable thresholds
}

/**
 * Training hyperparameters configuration.
 * Matches backend TrainingHyperparameters schema.
 */
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

  // Sparsity
  /** L1 sparsity penalty coefficient */
  l1_alpha: number;
  /** Target L0 sparsity (fraction of active features, 0-1) */
  target_l0?: number;
  /** Top-K sparsity percentage (e.g., 5 for 5%). Guarantees exact sparsity. */
  top_k_sparsity?: number;

  // JumpReLU-specific parameters (Gemma Scope architecture)
  /** Initial threshold value for JumpReLU activation (default: 0.001) */
  initial_threshold?: number;
  /** KDE bandwidth for STE gradient estimation in JumpReLU (default: 0.001) */
  bandwidth?: number;
  /** L0 sparsity coefficient for JumpReLU (default: 6e-4). Overrides l1_alpha for JumpReLU. */
  sparsity_coeff?: number;
  /** Whether to normalize decoder columns to unit norm (required for JumpReLU) */
  normalize_decoder?: boolean;

  // Training
  /** Initial learning rate */
  learning_rate: number;
  /** Training batch size */
  batch_size: number;
  /** Total training steps */
  total_steps: number;
  /** Linear warmup steps */
  warmup_steps?: number;

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
  /** Dataset ID for training data */
  dataset_id: string;
  /** Activation extraction ID (if using pre-extracted activations) */
  extraction_id?: string | null;

  // Status and progress
  /** Current training status */
  status: TrainingStatus;
  /** Training progress (0-100) */
  progress: number;
  /** Current training step */
  current_step: number;
  /** Total planned training steps */
  total_steps: number;

  // Hyperparameters
  /** Training hyperparameters */
  hyperparameters: HyperparametersConfig;

  // Current metrics (latest values)
  /** Current reconstruction loss */
  current_loss?: number | null;
  /** Current L0 sparsity (fraction of active features) */
  current_l0_sparsity?: number | null;
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
  /** Dataset ID for training data */
  dataset_id: string;
  /** Activation extraction ID (optional) */
  extraction_id?: string;
  /** Training hyperparameters */
  hyperparameters: HyperparametersConfig;
}

/**
 * Training control request.
 * Matches backend TrainingControlRequest schema.
 */
export interface TrainingControlRequest {
  /** Control action to perform */
  action: 'pause' | 'resume' | 'stop';
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
  l0_sparsity: number;
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
