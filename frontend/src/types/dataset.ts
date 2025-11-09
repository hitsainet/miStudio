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
  /** Padding strategy used (e.g., 'max_length', 'longest', 'do_not_pad') */
  padding?: string;
  /** Truncation strategy used (e.g., 'longest_first', 'only_first', 'only_second', 'do_not_truncate') */
  truncation?: string;
  /** Total number of tokens across all samples */
  num_tokens: number;
  /** Average sequence length in tokens */
  avg_seq_length: number;
  /** Minimum sequence length in tokens */
  min_seq_length: number;
  /** Maximum sequence length in tokens */
  max_seq_length: number;
  /** Median sequence length in tokens */
  median_seq_length?: number;
  /** Number of unique tokens in the tokenized dataset (vocabulary size) */
  vocab_size?: number;
  /** Distribution of sequence lengths bucketed by range (e.g., '0-100': 150) */
  length_distribution?: Record<string, number>;
  /** Distribution of samples across splits (e.g., {'train': 8000, 'validation': 1500, 'test': 500}) */
  split_distribution?: Record<string, number>;
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

export enum TokenizationStatus {
  QUEUED = 'queued',
  PROCESSING = 'processing',
  READY = 'ready',
  ERROR = 'error',
}

export interface DatasetTokenization {
  id: string;
  dataset_id: string;
  model_id: string;
  tokenized_path?: string;
  tokenizer_repo_id: string;
  vocab_size?: number;
  num_tokens?: number;
  avg_seq_length?: number;
  status: TokenizationStatus | 'queued' | 'processing' | 'ready' | 'error';
  progress?: number;
  error_message?: string;
  celery_task_id?: string;
  created_at: string;
  updated_at: string;
  completed_at?: string;
}

export interface DatasetTokenizationListResponse {
  data: DatasetTokenization[];
  total: number;
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
  num_samples?: number;
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
