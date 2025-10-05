/**
 * Model Types
 *
 * Backend API Contract:
 * - GET /api/models - List all models
 * - GET /api/models/:id - Get model details with architecture
 * - POST /api/models/download - Download model (body: { repoId, quantization, accessToken? })
 * - DELETE /api/models/:id - Delete model
 * - GET /api/models/:id/architecture - Get architecture details
 * - POST /api/models/:id/extract-activations - Extract activations (long-running job)
 *
 * Quantization options: 'Q4', 'Q8', 'FP16', 'FP32', 'Q2'
 * Status transitions: downloading -> loading -> quantizing -> ready
 */

/**
 * Model representation with quantization support
 */
export interface Model {
  id: string;
  name: string;
  params: string; // Human-readable param count (e.g., '1.1B', '135M')
  quantized: 'FP32' | 'FP16' | 'Q8' | 'Q4' | 'Q2';
  mem_req: string; // Memory requirement (e.g., '1.2GB')
  status: 'downloading' | 'loading' | 'quantizing' | 'ready' | 'error';
  progress?: number; // 0-100, present during downloading
  error?: string; // Error message if status is 'error'
  created_at?: string; // ISO 8601 timestamp
}

/**
 * Model architecture details
 *
 * Backend should extract from transformers model config
 */
export interface ModelArchitecture {
  architecture: string; // e.g., 'LlamaForCausalLM', 'GPT2LMHeadModel'
  hidden_size: number; // d_model
  num_layers: number;
  num_attention_heads: number;
  intermediate_size: number; // FFN hidden size
  vocab_size: number;
  layers: LayerInfo[];
}

/**
 * Individual layer information for hook point selection
 */
export interface LayerInfo {
  index: number;
  name: string; // e.g., 'transformer.h.0'
  type: string; // e.g., 'TransformerBlock', 'Embedding'
}

/**
 * Activation extraction configuration
 *
 * Backend API Contract:
 * - POST /api/models/:id/extract-activations
 * Returns job_id for polling status
 *
 * Implementation:
 * - Register forward hooks on specified layers
 * - Extract residual stream, MLP outputs, or attention outputs
 * - Store as memory-mapped tensors: [n_samples, seq_len, hidden_dim]
 * - Use PyTorch hooks or TransformerLens library
 */
export interface ActivationExtractionConfig {
  dataset_id: string;
  layers: number[]; // Which transformer layers (e.g., [0, 4, 8, 12])
  activation_types: ('residual' | 'mlp' | 'attention')[]; // Which activation types
  batch_size: number;
  max_samples?: number; // Optional limit for testing
}

/**
 * Activation extraction job status
 */
export interface ActivationExtractionStatus {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'error';
  progress: number; // 0-100
  estimated_duration_seconds?: number;
  error?: string;
}

/**
 * Model download request parameters
 */
export interface ModelDownloadRequest {
  repo_id: string;
  quantization: 'FP32' | 'FP16' | 'Q8' | 'Q4' | 'Q2';
  hf_token?: string; // Optional for gated models
}
