/**
 * Dataset Types
 *
 * Backend API Contract:
 * - GET /api/datasets - List all datasets
 * - GET /api/datasets/:id - Get dataset details
 * - POST /api/datasets/download - Download from HuggingFace (body: { repoId, accessToken? })
 * - DELETE /api/datasets/:id - Delete dataset
 * - GET /api/datasets/:id/statistics - Get statistics
 * - GET /api/datasets/:id/samples - Browse samples with pagination
 * - POST /api/datasets/:id/tokenize - Tokenize dataset
 *
 * Status transitions: downloading -> ingesting -> ready
 * Real-time updates: Poll GET /api/datasets/:id for status updates every 2-5 seconds
 */

/**
 * Dataset representation from HuggingFace or other sources
 */
export interface Dataset {
  id: string;
  name: string;
  source: 'HuggingFace' | 'Local' | 'Custom';
  size: string; // Human-readable size (e.g., '2.3GB')
  status: 'downloading' | 'ingesting' | 'ready' | 'error';
  status_message?: string; // Human-readable status description from backend
  progress?: number; // 0-100, present during downloading/ingesting
  error?: string; // Error message if status is 'error'
  created_at?: string; // ISO 8601 timestamp
}

/**
 * Dataset sample/text for browsing and analysis
 *
 * Backend API Contract:
 * - GET /api/datasets/:id/samples?page=1&limit=50&split=train&search=query
 * - Pagination required for large datasets
 * - Full-text search should use database FTS or Elasticsearch
 */
export interface DatasetSample {
  id: number;
  text: string;
  split: 'train' | 'validation' | 'test';
  metadata?: Record<string, any>; // Source, domain, etc.
  tokens?: string[]; // Tokenized representation
  token_count?: number;
}

/**
 * Dataset statistics for overview
 *
 * Computed during ingestion and cached
 * Backend should compute during ingestion and store in database
 */
export interface DatasetStatistics {
  total_samples: number;
  total_tokens: number;
  avg_tokens_per_sample: number;
  unique_tokens: number;
  min_length: number;
  median_length: number;
  max_length: number;
  length_distribution?: DistributionBucket[];
}

/**
 * Distribution bucket for visualizations (histograms)
 */
export interface DistributionBucket {
  range: string; // e.g., '100-200'
  count: number;
}

/**
 * Tokenization settings for dataset processing
 *
 * Backend API Contract:
 * - POST /api/datasets/:id/tokenize
 * Backend: Should match model's tokenizer exactly
 * Tokenizer loading: Use HuggingFace transformers library
 */
export interface TokenizationSettings {
  tokenizer?: string; // 'auto' to use model's tokenizer
  max_length: number; // Max sequence length
  truncation: boolean;
  padding: 'max_length' | 'longest' | 'do_not_pad';
  add_special_tokens: boolean;
  return_attention_mask?: boolean;
}

/**
 * Dataset download request parameters
 */
export interface DatasetDownloadRequest {
  repo_id: string;
  hf_token?: string; // Optional for gated datasets
  split?: 'train' | 'validation' | 'test' | 'all';
}

/**
 * Dataset samples query parameters
 */
export interface DatasetSamplesQuery {
  page?: number;
  limit?: number;
  search?: string;
  split?: 'train' | 'validation' | 'test';
}
