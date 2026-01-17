/**
 * SAE (Sparse Autoencoder) Types
 *
 * TypeScript interfaces for SAE Management feature.
 * Matches backend Pydantic schemas in src/schemas/sae.py
 *
 * Backend API Contract:
 * - GET /api/v1/saes - List all SAEs
 * - GET /api/v1/saes/:id - Get SAE details
 * - POST /api/v1/saes/hf/preview - Preview HuggingFace repository
 * - POST /api/v1/saes/download - Download SAE from HuggingFace
 * - POST /api/v1/saes/upload - Upload SAE to HuggingFace
 * - POST /api/v1/saes/import/training - Import SAE from training
 * - POST /api/v1/saes/import/file - Import SAE from local file
 * - DELETE /api/v1/saes/:id - Delete SAE
 * - POST /api/v1/saes/delete - Batch delete SAEs
 * - GET /api/v1/saes/:id/features - Browse features for steering
 */

/**
 * SAE source types.
 * Matches backend SAESource enum.
 */
export enum SAESource {
  HUGGINGFACE = 'huggingface',
  LOCAL = 'local',
  TRAINED = 'trained',
}

/**
 * SAE processing status.
 * Matches backend SAEStatus enum.
 */
export enum SAEStatus {
  PENDING = 'pending',
  DOWNLOADING = 'downloading',
  CONVERTING = 'converting',
  READY = 'ready',
  ERROR = 'error',
  DELETED = 'deleted',
}

/**
 * SAE file formats.
 * Matches backend SAEFormat enum.
 */
export enum SAEFormat {
  SAELENS = 'saelens',
  MISTUDIO = 'mistudio',
  CUSTOM = 'custom',
}

/**
 * File info from HuggingFace repository.
 */
export interface HFFileInfo {
  filepath: string;
  size_bytes: number;
  is_sae: boolean;
  layer?: number;
  n_features?: number;
}

/**
 * HuggingFace repository preview response.
 */
export interface HFRepoPreviewResponse {
  repo_id: string;
  repo_type: string;
  description: string | null;
  files: HFFileInfo[];
  sae_files: HFFileInfo[];
  sae_paths: string[];
  model_name: string | null;
  total_size_bytes: number | null;
}

/**
 * Request to preview a HuggingFace repository.
 */
export interface HFRepoPreviewRequest {
  repo_id: string;
  access_token?: string;
}

/**
 * Request to download an SAE from HuggingFace.
 */
export interface SAEDownloadRequest {
  repo_id: string;
  filepath: string;
  name?: string;
  description?: string;
  revision?: string;
  access_token?: string;
  model_name?: string;
  /** ID of local model to link for steering (from Models panel) */
  model_id?: string;
}

/**
 * Request to upload an SAE to HuggingFace.
 */
export interface SAEUploadRequest {
  sae_id: string;
  repo_id: string;
  filepath: string;
  access_token: string;
  create_repo?: boolean;
  private?: boolean;
  commit_message?: string;
}

/**
 * Response from SAE upload.
 */
export interface SAEUploadResponse {
  sae_id: string;
  repo_id: string;
  filepath: string;
  url: string;
  commit_hash: string | null;
}

/**
 * Request to import SAE(s) from a completed training job.
 * Supports importing multiple SAEs from multi-layer/multi-hook trainings.
 */
export interface SAEImportFromTrainingRequest {
  training_id: string;
  name?: string;
  description?: string;
  /** Import all available SAEs (default: true) */
  import_all?: boolean;
  /** Specific layers to import (if import_all=false) */
  layers?: number[];
  /** Specific hook types to import (if import_all=false) */
  hook_types?: string[];
}

/**
 * Info about an available SAE in a training checkpoint.
 */
export interface AvailableSAEInfo {
  layer: number;
  hook_type: string;
  path: string;
  size_bytes: number | null;
}

/**
 * Response listing available SAEs in a completed training.
 */
export interface TrainingAvailableSAEsResponse {
  training_id: string;
  available_saes: AvailableSAEInfo[];
  total_count: number;
}

/**
 * Response from importing SAEs from training.
 */
export interface SAEImportFromTrainingResponse {
  imported_count: number;
  sae_ids: string[];
  saes: SAE[];
  training_id: string;
  message: string;
}

/**
 * Request to import SAE from a local file.
 */
export interface SAEImportFromFileRequest {
  file_path: string;
  name: string;
  description?: string;
  format?: SAEFormat;
  model_name?: string;
  layer?: number;
}

/**
 * SAE metadata stored in sae_metadata JSONB field.
 */
export interface SAEMetadata {
  files_downloaded?: string[];
  is_directory?: boolean;
  training_hyperparameters?: Record<string, any>;
  training_status?: string;
  final_loss?: number;
  final_l0_sparsity?: number;
  original_path?: string;
  activation_stats?: Record<string, any>;
  neuronpedia_url?: string;
  [key: string]: any;
}

/**
 * Full SAE response from API.
 */
export interface SAE {
  id: string;
  name: string;
  description: string | null;
  source: SAESource;
  status: SAEStatus;

  // HuggingFace source info
  hf_repo_id: string | null;
  hf_filepath: string | null;
  hf_revision: string | null;

  // Training source info
  training_id: string | null;

  // Model compatibility
  model_name: string | null;
  model_id: string | null;

  // SAE architecture info
  layer: number | null;
  hook_type: string | null;
  n_features: number | null;
  d_model: number | null;
  architecture: string | null;

  // Format and storage
  format: SAEFormat;
  local_path: string | null;
  file_size_bytes: number | null;

  // Progress and status
  progress: number;
  error_message: string | null;

  // Metadata
  sae_metadata: SAEMetadata;

  // Timestamps
  created_at: string;
  updated_at: string;
  downloaded_at: string | null;
}

/**
 * Paginated list of SAEs.
 */
export interface SAEListResponse {
  data: SAE[];
  pagination: {
    skip: number;
    limit: number;
    total: number;
    has_more: boolean;
  };
}

/**
 * SAE download progress update (for WebSocket).
 */
export interface SAEDownloadProgress {
  sae_id: string;
  status: SAEStatus;
  progress: number;
  bytes_downloaded: number | null;
  total_bytes: number | null;
  speed_bytes_per_sec: number | null;
}

/**
 * SAE feature summary for feature browser.
 */
export interface SAEFeatureSummary {
  feature_idx: number;
  layer: number;
  label: string | null;
  activation_count?: number;
  mean_activation?: number;
  max_activation: number | null;
  top_tokens: string[];
  neuronpedia_url: string | null;
  feature_id: string | null;
}

/**
 * Feature browser response.
 */
export interface SAEFeatureBrowserResponse {
  sae_id: string;
  n_features: number;
  features: SAEFeatureSummary[];
  pagination: {
    skip: number;
    limit: number;
    total: number;
    has_more: boolean;
  };
}

/**
 * Batch delete request.
 */
export interface SAEDeleteRequest {
  ids: string[];
}

/**
 * Batch delete response.
 */
export interface SAEDeleteResponse {
  deleted_count: number;
  failed_count: number;
  deleted_ids: string[];
  failed_ids: string[];
  errors: Record<string, string>;
  message: string;
}
