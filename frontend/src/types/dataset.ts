export enum DatasetStatus {
  DOWNLOADING = 'downloading',
  PROCESSING = 'processing',
  READY = 'ready',
  ERROR = 'error',
}

/**
 * Dataset schema information describing column structure.
 * Matches backend Pydantic SchemaMetadata schema.
 */
export interface SchemaMetadata {
  /** List of text/string columns in the dataset */
  text_columns: string[];
  /** Mapping of column names to their data types */
  column_info: Record<string, string>;
  /** Complete list of all columns in the dataset */
  all_columns: string[];
  /** Whether the dataset has multiple text columns */
  is_multi_column: boolean;
}

/**
 * Tokenization statistics and configuration.
 * Matches backend Pydantic TokenizationMetadata schema.
 */
export interface TokenizationMetadata {
  /** HuggingFace tokenizer identifier (e.g., 'gpt2', 'bert-base-uncased') */
  tokenizer_name: string;
  /** Name of the text column that was tokenized */
  text_column_used: string;
  /** Maximum sequence length in tokens (1-8192) */
  max_length: number;
  /** Sliding window stride (0 = no overlap) */
  stride: number;
  /** Total number of tokens across all samples */
  num_tokens: number;
  /** Average sequence length in tokens */
  avg_seq_length: number;
  /** Minimum sequence length in tokens */
  min_seq_length: number;
  /** Maximum sequence length in tokens */
  max_seq_length: number;
}

/**
 * Dataset download metadata.
 * Matches backend Pydantic DownloadMetadata schema.
 */
export interface DownloadMetadata {
  /** Dataset split downloaded (e.g., 'train', 'validation', 'test') */
  split?: string;
  /** Dataset configuration name (e.g., 'en', 'zh') for multi-config datasets */
  config?: string;
  /** Whether an access token was used for download */
  access_token_provided?: boolean;
}

/**
 * Complete dataset metadata structure.
 * Matches backend Pydantic DatasetMetadata schema.
 *
 * Note: Backend uses 'schema' key but it maps to 'dataset_schema' internally.
 * Frontend receives it as 'schema' from API.
 */
export interface DatasetMetadata {
  /** Dataset schema information */
  schema?: SchemaMetadata;
  /** Tokenization statistics and configuration */
  tokenization?: TokenizationMetadata;
  /** Download metadata (split, config, etc.) */
  download?: DownloadMetadata;
}

export interface Dataset {
  id: string;
  name: string;
  source: string;
  hf_repo_id?: string;
  status: DatasetStatus | 'downloading' | 'processing' | 'ready' | 'error';
  progress?: number;
  error_message?: string;
  raw_path?: string;
  tokenized_path?: string;
  num_samples?: number;
  num_tokens?: number;
  avg_seq_length?: number;
  vocab_size?: number;
  size_bytes?: number;
  /** Structured dataset metadata with schema, tokenization, and download info */
  metadata?: DatasetMetadata;
  created_at: string;
  updated_at: string;
}

export interface DatasetSample {
  id: number;
  text: string;
  split: 'train' | 'validation' | 'test';
  metadata?: Record<string, any>;
  tokens?: string[];
  token_count?: number;
}

export interface DatasetStatistics {
  total_samples: number;
  total_tokens: number;
  avg_tokens_per_sample: number;
  unique_tokens: number;
  min_length: number;
  median_length: number;
  max_length: number;
  distribution?: Array<{
    range: string;
    count: number;
  }>;
}

export interface DatasetDownloadRequest {
  repo_id: string;
  access_token?: string;
  split?: string;
  config?: string;
}

export interface DatasetListResponse {
  data: Dataset[];
  pagination?: {
    page: number;
    total: number;
    has_next: boolean;
  };
}

export interface DatasetResponse {
  data: Dataset;
}
