export enum DatasetStatus {
  DOWNLOADING = 'downloading',
  PROCESSING = 'processing',
  READY = 'ready',
  ERROR = 'error',
}

export interface Dataset {
  id: string;
  name: string;
  source: string;
  hf_repo_id?: string;
  status: DatasetStatus;
  progress?: number;
  error_message?: string;
  raw_path?: string;
  tokenized_path?: string;
  num_samples?: number;
  num_tokens?: number;
  avg_seq_length?: number;
  vocab_size?: number;
  size_bytes?: number;
  metadata?: Record<string, any>;
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
