/**
 * Model Types
 *
 * TypeScript interfaces for Model Management feature.
 * Matches backend Pydantic schemas in src/schemas/model.py and src/models/model.py
 * Aligned with Mock UI reference: 0xcc/project-specs/reference-implementation/src/types/model.types.ts
 *
 * Backend API Contract:
 * - GET /api/v1/models - List all models
 * - GET /api/v1/models/:id - Get model details with architecture
 * - POST /api/v1/models/download - Download model (body: { repo_id, quantization, access_token? })
 * - DELETE /api/v1/models/:id - Delete model
 * - GET /api/v1/models/:id/architecture - Get architecture details
 * - POST /api/v1/models/:id/extract-activations - Extract activations (long-running job)
 *
 * Quantization options: 'Q4', 'Q8', 'FP16', 'FP32', 'Q2'
 * Status transitions: downloading -> loading -> quantizing -> ready (or error)
 */

/**
 * Quantization formats supported by the system.
 * Matches backend QuantizationFormat enum.
 */
export enum QuantizationFormat {
  FP32 = 'FP32',
  FP16 = 'FP16',
  Q8 = 'Q8',
  Q4 = 'Q4',
  Q2 = 'Q2',
}

/**
 * Model processing status.
 * Matches backend ModelStatus enum.
 */
export enum ModelStatus {
  DOWNLOADING = 'downloading',
  LOADING = 'loading',
  QUANTIZING = 'quantizing',
  READY = 'ready',
  ERROR = 'error',
}

/**
 * Model architecture configuration extracted from model metadata.
 * Contains structural information about the model's layers and dimensions.
 * Used internally - maps to backend architecture_config JSONB field.
 */
export interface ArchitectureConfig {
  /** Number of hidden layers in the model */
  num_hidden_layers: number;
  /** Hidden dimension size */
  hidden_size: number;
  /** Number of attention heads */
  num_attention_heads: number;
  /** Intermediate/feed-forward dimension size */
  intermediate_size?: number;
  /** Vocabulary size */
  vocab_size?: number;
  /** Maximum position embeddings (context length) */
  max_position_embeddings?: number;
  /** Number of key-value heads for grouped query attention (GQA) */
  num_key_value_heads?: number;
  /** RoPE theta parameter for rotary embeddings */
  rope_theta?: number;
  /** Additional architecture-specific parameters */
  [key: string]: any;
}

/**
 * Model architecture details for the Architecture Viewer modal.
 * Backend should extract from transformers model config.
 * Matches Mock UI reference ModelArchitecture interface.
 */
export interface ModelArchitecture {
  /** Architecture name (e.g., 'LlamaForCausalLM', 'GPT2LMHeadModel') */
  architecture: string;
  /** Hidden dimension size (d_model) */
  hidden_size: number;
  /** Number of transformer layers */
  num_layers: number;
  /** Number of attention heads */
  num_attention_heads: number;
  /** FFN intermediate hidden size */
  intermediate_size: number;
  /** Vocabulary size */
  vocab_size: number;
  /** List of layer information for hook point selection */
  layers: LayerInfo[];
}

/**
 * Individual layer information for hook point selection.
 * Used in Architecture Viewer to display layer structure.
 * Matches Mock UI reference LayerInfo interface.
 */
export interface LayerInfo {
  /** Layer index (0-based) */
  index: number;
  /** Layer name (e.g., 'transformer.h.0', 'model.layers.0') */
  name: string;
  /** Layer type (e.g., 'TransformerBlock', 'Embedding', 'LayerNorm') */
  type: string;
}

/**
 * Configuration for activation extraction.
 * Defines which layers and hook types to use for extraction.
 * Matches Mock UI reference ActivationExtractionConfig interface.
 *
 * Backend API Contract:
 * - POST /api/v1/models/:id/extract-activations
 * Returns job_id for polling status
 *
 * Implementation:
 * - Register forward hooks on specified layers
 * - Extract residual stream, MLP outputs, or attention outputs
 * - Store as memory-mapped tensors: [n_samples, seq_len, hidden_dim]
 * - Use PyTorch hooks (via HookManager in backend/src/ml/forward_hooks.py)
 */
export interface ActivationExtractionConfig {
  /** Dataset ID to use for extraction */
  dataset_id: string;
  /** List of layer indices to extract from (e.g., [0, 5, 10, 15]) */
  layer_indices: number[];
  /**
   * Types of activations to extract ('residual', 'mlp', 'attention')
   * Alias: Can also use 'layers' in Mock UI reference
   */
  hook_types: ('residual' | 'mlp' | 'attention')[];
  /** Maximum number of samples to process (optional limit for testing) */
  max_samples: number;
  /** Batch size for processing */
  batch_size?: number;
  /** GPU micro-batch size for memory efficiency (defaults to batch_size if not specified) */
  micro_batch_size?: number;
  /** Top K examples to save per feature */
  top_k_examples?: number;
}

/**
 * Activation extraction job status.
 * Used for polling extraction progress.
 * Matches Mock UI reference ActivationExtractionStatus interface.
 */
export interface ActivationExtractionStatus {
  /** Job ID for tracking */
  job_id: string;
  /** Current job status */
  status: 'queued' | 'processing' | 'completed' | 'error';
  /** Progress percentage (0-100) */
  progress: number;
  /** Estimated duration in seconds */
  estimated_duration_seconds?: number;
  /** Error message if status is 'error' */
  error?: string;
}

/**
 * Activation extraction result metadata.
 */
export interface ActivationExtractionResult {
  /** Unique extraction ID */
  extraction_id: string;
  /** Path to extraction output directory */
  output_path: string;
  /** Number of samples processed */
  num_samples: number;
  /** List of saved activation files */
  saved_files: string[];
  /** Activation statistics per layer */
  statistics: Record<string, ActivationStatistics>;
  /** Path to metadata file */
  metadata_path: string;
}

/**
 * Statistics calculated for extracted activations.
 */
export interface ActivationStatistics {
  /** Shape of activation tensor [num_samples, seq_len, hidden_dim] */
  shape: number[];
  /** Mean magnitude of activations */
  mean_magnitude: number;
  /** Maximum activation value */
  max_activation: number;
  /** Minimum activation value */
  min_activation: number;
  /** Standard deviation of activations */
  std_activation: number;
  /** Percentage of near-zero activations (sparsity) */
  sparsity_percent: number;
  /** Size in megabytes */
  size_mb: number;
}

/**
 * Main model interface.
 * Matches backend ModelResponse schema.
 */
export interface Model {
  /** Unique model identifier (format: m_{uuid}) */
  id: string;
  /** Model name */
  name: string;
  /** HuggingFace repository ID (e.g., 'TinyLlama/TinyLlama-1.1B') */
  repo_id?: string;
  /** Model architecture (llama, gpt2, phi, etc.) */
  architecture: string;
  /** Number of parameters */
  params_count: number;
  /** Quantization format */
  quantization: QuantizationFormat;
  /** Current processing status */
  status: ModelStatus | 'downloading' | 'loading' | 'quantizing' | 'ready' | 'error';
  /** Download/loading progress (0-100) */
  progress?: number;
  /** Error message if status is ERROR */
  error_message?: string;
  /** Path to raw model files */
  file_path?: string;
  /** Path to quantized model files */
  quantized_path?: string;
  /** Architecture configuration */
  architecture_config?: ArchitectureConfig;
  /** Estimated memory requirement in bytes */
  memory_required_bytes?: number;
  /** Disk size in bytes */
  disk_size_bytes?: number;
  /** Whether model has any completed extraction jobs */
  has_completed_extractions?: boolean;
  /** Record creation timestamp */
  created_at: string;
  /** Record last update timestamp */
  updated_at: string;
  /** Current extraction ID (if extraction is running) */
  extraction_id?: string;
  /** Extraction progress (0-100) */
  extraction_progress?: number;
  /** Extraction status */
  extraction_status?: 'starting' | 'loading' | 'extracting' | 'saving' | 'complete' | 'failed' | 'error';
  /** Extraction message */
  extraction_message?: string;
  /** Extraction error type (OOM, VALIDATION, TIMEOUT, EXTRACTION, UNKNOWN) */
  extraction_error_type?: string;
  /** Suggested retry parameters for failed extraction */
  extraction_suggested_retry_params?: Record<string, any>;
}

/**
 * Request to download a model from HuggingFace.
 */
export interface ModelDownloadRequest {
  /** HuggingFace repository ID (e.g., 'TinyLlama/TinyLlama-1.1B') */
  repo_id: string;
  /** Quantization format to apply */
  quantization: QuantizationFormat;
  /** HuggingFace access token for gated models */
  access_token?: string;
}

/**
 * Request to extract activations from a model.
 */
export interface ActivationExtractionRequest {
  /** Model ID */
  model_id: string;
  /** Extraction configuration */
  config: ActivationExtractionConfig;
}

/**
 * Paginated list of models response.
 */
export interface ModelListResponse {
  /** List of models */
  data: Model[];
  /** Pagination metadata */
  pagination?: {
    /** Current page number */
    page: number;
    /** Total number of models */
    total: number;
    /** Whether there are more pages */
    has_next: boolean;
  };
}

/**
 * Single model response.
 */
export interface ModelResponse {
  /** Model data */
  data: Model;
}

/**
 * WebSocket progress event for model operations.
 */
export interface ModelProgressEvent {
  /** Event type */
  type: 'progress' | 'completed' | 'error' | 'extraction_progress' | 'extraction_completed';
  /** Model ID */
  model_id: string;
  /** Progress percentage (0-100) */
  progress: number;
  /** Current status */
  status: ModelStatus | string;
  /** Progress message */
  message?: string;
  /** Download speed in Mbps */
  speed_mbps?: number;
  /** GPU utilization percentage */
  gpu_utilization?: number;
  /** Samples processed (for extraction) */
  samples_processed?: number;
  /** Estimated time to completion in seconds */
  eta_seconds?: number;
  /** Error message (for error events) */
  error?: string;
  /** Suggested fallback quantization format (for OOM errors) */
  suggested_format?: QuantizationFormat;
  /** Extraction result (for extraction_completed events) */
  extraction_result?: ActivationExtractionResult;
}
