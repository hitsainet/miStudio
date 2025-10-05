/**
 * Training Types
 *
 * Backend API Contract:
 * - POST /api/training/start - Start training (body: TrainingConfig)
 * - POST /api/training/:id/pause - Pause training (idempotent)
 * - POST /api/training/:id/resume - Resume training (idempotent)
 * - POST /api/training/:id/stop - Stop training permanently
 * - GET /api/training/:id/status - Get current status
 * - WebSocket /ws/training/:id - Real-time progress updates
 * - GET /api/training/:id/logs - Stream logs via SSE
 * - GET /api/training/:id/checkpoints - List checkpoints
 * - POST /api/training/:id/checkpoint - Save checkpoint
 * - POST /api/training/:id/checkpoint/:cpId/load - Load checkpoint
 *
 * Status transitions:
 * initializing -> training -> completed
 *                  |  ^
 *                  v  |
 *                paused
 *                  |
 *                  v
 *                stopped
 *
 * Progress updates should stream via WebSocket with:
 * { progress: number, metrics: TrainingMetrics, timestamp: string }
 */

/**
 * Training hyperparameters for SAE (Sparse Autoencoder) training
 *
 * Backend Validation:
 * - learningRate: 1e-6 to 1e-2
 * - batchSize: Power of 2, typically 32-512
 * - l1Coefficient: 1e-5 to 1e-1 (sparsity penalty)
 * - expansionFactor: 1-32 (hidden layer expansion)
 * - trainingSteps: 1000-1000000
 */
export interface Hyperparameters {
  learning_rate: number;
  batch_size: number;
  l1_coefficient: number; // λ for L1 sparsity penalty
  expansion_factor: number; // d_sae = d_model * expansion_factor
  training_steps: number;
  optimizer: 'AdamW' | 'Adam' | 'SGD';
  lr_schedule: 'constant' | 'cosine' | 'linear' | 'exponential';
  ghost_grad_penalty: boolean; // Enable ghost gradient penalty for dead neurons
}

/**
 * Training metrics tracked during SAE training
 *
 * Backend should compute and publish via WebSocket every N steps
 */
export interface TrainingMetrics {
  loss: number | null;
  reconstruction_loss?: number; // MSE loss
  sparsity_loss?: number; // L1 penalty
  l0_sparsity: number | null; // Average active features (important!)
  dead_neurons?: number; // Neurons that never activate
  explained_variance?: number; // How well SAE reconstructs original
  elapsed_seconds?: number;
  estimated_remaining_seconds?: number; // Backend calculates from current rate
  rate?: number; // Steps per second
}

/**
 * Training job configuration and state
 */
export interface Training {
  id: string;
  model_id: string;
  dataset_id: string;
  encoder_type: 'sparse' | 'skip' | 'transcoder'; // SAE architecture type
  status: 'initializing' | 'training' | 'paused' | 'stopped' | 'completed' | 'error';
  progress: number; // 0-100
  start_time: string; // ISO 8601 timestamp
  end_time?: string; // ISO 8601 timestamp
  hyperparameters: Hyperparameters;
  metrics: TrainingMetrics;
  error?: string;
}

/**
 * Training checkpoint for resuming or rollback
 *
 * Backend API Contract:
 * - GET /api/training/:trainingId/checkpoints - List checkpoints
 * - POST /api/training/:trainingId/checkpoint - Save checkpoint
 * - POST /api/training/:trainingId/checkpoint/:id/load - Load checkpoint
 * - DELETE /api/training/:trainingId/checkpoint/:id - Delete checkpoint
 *
 * Storage: Checkpoints should be stored in local file storage
 * with metadata in database for quick retrieval
 */
export interface Checkpoint {
  id: string;
  training_id: string;
  step: number; // Training step number
  loss: number;
  timestamp: string; // ISO 8601
  storage_url?: string; // URL to checkpoint file in object storage
  size?: string; // File size (human-readable)
}

/**
 * Training start request configuration
 */
export interface TrainingStartRequest {
  model_id: string;
  dataset_id: string;
  encoder_type: 'sparse' | 'skip' | 'transcoder';
  hyperparameters: Hyperparameters;
}

/**
 * Training log entry for streaming
 *
 * Backend streams via Server-Sent Events (SSE)
 */
export interface TrainingLogEntry {
  level: 'debug' | 'info' | 'warning' | 'error';
  message: string;
  timestamp: string; // ISO 8601
  step?: number;
}

/**
 * WebSocket message types for training updates
 */
export type TrainingWebSocketMessage =
  | { type: 'training.progress'; data: { progress: number; metrics: TrainingMetrics } }
  | { type: 'training.completed'; data: { training_id: string } }
  | { type: 'training.error'; data: { error: string } }
  | { type: 'metrics.update'; data: TrainingMetrics };
