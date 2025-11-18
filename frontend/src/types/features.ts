/**
 * Feature Discovery Types
 *
 * TypeScript interfaces for feature extraction and discovery functionality.
 * Matches backend API contracts for feature discovery endpoints.
 */

/**
 * Extraction job status.
 */
export type ExtractionStatus = 'queued' | 'extracting' | 'completed' | 'failed' | 'cancelled';

/**
 * Feature label source.
 */
export type LabelSource = 'auto' | 'user';

/**
 * Extraction configuration request.
 */
export interface ExtractionConfigRequest {
  evaluation_samples: number;  // 1,000 - 100,000
  top_k_examples: number;      // 10 - 1,000
  // Token filtering configuration (matches labeling filter structure)
  filter_special?: boolean;     // Filter special tokens (<s>, </s>, etc.) (default: true)
  filter_single_char?: boolean; // Filter single character tokens (default: true)
  filter_punctuation?: boolean; // Filter pure punctuation (default: true)
  filter_numbers?: boolean;     // Filter pure numeric tokens (default: true)
  filter_fragments?: boolean;   // Filter word fragments (BPE subwords) (default: true)
  filter_stop_words?: boolean;  // Filter common stop words (the, and, is, etc.) (default: false)
  vectorization_batch_size?: string | number;  // 'auto' or 1-256 (default: 'auto')
  soft_time_limit?: number;  // Soft time limit in seconds (default: 144000 = 40 hours)
  time_limit?: number;  // Hard time limit in seconds (default: 172800 = 48 hours)
}

/**
 * Extraction job status response.
 */
export interface ExtractionStatusResponse {
  id: string;
  training_id: string;
  model_name: string | null;
  dataset_name: string | null;
  status: ExtractionStatus;
  progress: number | null;
  features_extracted: number | null;
  total_features: number | null;
  error_message: string | null;
  config: Record<string, any>;
  statistics: ExtractionStatistics | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  // Token filtering configuration (matches labeling filter structure)
  filter_special?: boolean;
  filter_single_char?: boolean;
  filter_punctuation?: boolean;
  filter_numbers?: boolean;
  filter_fragments?: boolean;
  filter_stop_words?: boolean;
}

/**
 * Extraction statistics.
 */
export interface ExtractionStatistics {
  total_features: number;
  interpretable_count: number;
  avg_activation_frequency: number;
  avg_interpretability: number;
}

/**
 * Feature activation example.
 */
export interface FeatureActivationExample {
  tokens: string[];
  activations: number[];
  max_activation: number;
  sample_index: number;
}

/**
 * Feature response.
 */
export interface Feature {
  id: string;
  training_id: string;
  extraction_job_id: string;
  neuron_index: number;
  category: string | null;
  name: string;
  description: string | null;
  label_source: LabelSource;
  activation_frequency: number;
  interpretability_score: number;
  max_activation: number;
  mean_activation: number | null;
  is_favorite: boolean;
  notes: string | null;
  created_at: string;
  updated_at: string;
  example_context: FeatureActivationExample | null;
}

/**
 * Feature statistics.
 */
export interface FeatureStatistics {
  total_features: number;
  interpretable_percentage: number;
  avg_activation_frequency: number;
}

/**
 * Feature list response with pagination.
 */
export interface FeatureListResponse {
  features: Feature[];
  total: number;
  limit: number;
  offset: number;
  statistics: FeatureStatistics;
}

/**
 * Feature search request parameters.
 */
export interface FeatureSearchRequest {
  search?: string | null;
  sort_by?: 'activation_freq' | 'interpretability' | 'feature_id';
  sort_order?: 'asc' | 'desc';
  is_favorite?: boolean | null;
  limit?: number;
  offset?: number;
}

/**
 * Feature detail response.
 */
export interface FeatureDetail extends Feature {
  active_samples: number;
}

/**
 * Feature update request.
 */
export interface FeatureUpdateRequest {
  name?: string | null;
  description?: string | null;
  notes?: string | null;
}

/**
 * Logit lens analysis response.
 */
export interface LogitLensResponse {
  top_tokens: string[];
  probabilities: number[];
  interpretation: string;
  computed_at: string;
}

/**
 * Correlated feature.
 */
export interface CorrelatedFeature {
  feature_id: string;
  feature_name: string;
  correlation: number;
}

/**
 * Feature correlations response.
 */
export interface CorrelationsResponse {
  correlated_features: CorrelatedFeature[];
  computed_at: string;
}

/**
 * Ablation analysis response.
 */
export interface AblationResponse {
  perplexity_delta: number;
  impact_score: number;
  baseline_perplexity: number;
  ablated_perplexity: number;
  computed_at: string;
}

/**
 * Token analysis token entry.
 */
export interface TokenAnalysisToken {
  rank: number;
  token: string;
  count: number;
  percentage: number;
}

/**
 * Token analysis summary statistics.
 */
export interface TokenAnalysisSummary {
  total_examples: number;
  original_token_count: number;
  filtered_token_count: number;
  junk_removed: number;
  total_token_occurrences: number;
  filtered_token_occurrences: number;
  diversity_percent: number;
}

/**
 * Token analysis response.
 */
export interface TokenAnalysisResponse {
  summary: TokenAnalysisSummary;
  tokens: TokenAnalysisToken[];
}
