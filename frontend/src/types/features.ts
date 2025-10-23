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
}

/**
 * Extraction job status response.
 */
export interface ExtractionStatusResponse {
  id: string;
  training_id: string;
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
