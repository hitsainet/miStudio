/**
 * Feature Discovery Types
 *
 * TypeScript interfaces for feature extraction and discovery functionality.
 * Matches backend API contracts for feature discovery endpoints.
 */

/**
 * Extraction job status.
 */
export type ExtractionStatus = 'queued' | 'extracting' | 'finalizing' | 'completed' | 'failed' | 'cancelled' | 'deleting';

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
  // NLP configuration
  auto_nlp?: boolean;  // Automatically start NLP analysis after extraction (default: true)
}

/**
 * Extraction source type.
 */
export type ExtractionSourceType = 'training' | 'external_sae';

/**
 * NLP processing status.
 */
export type NlpStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled' | null;

/**
 * Extraction job status response.
 * Supports both training-based and external SAE-based extractions.
 */
export interface ExtractionStatusResponse {
  id: string;
  // Source identification - exactly one will be set
  training_id: string | null;
  external_sae_id: string | null;
  source_type: ExtractionSourceType;
  // Display info
  model_name: string | null;
  dataset_name: string | null;
  sae_name: string | null;  // For external SAE sources
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
  // Context window configuration
  context_prefix_tokens?: number;
  context_suffix_tokens?: number;
  // NLP configuration
  auto_nlp?: boolean;  // Whether NLP was/will be auto-triggered after extraction
  // NLP Processing status (separate from feature extraction)
  nlp_status?: NlpStatus;
  nlp_progress?: number | null;
  nlp_processed_count?: number | null;
  nlp_error_message?: string | null;
  // Real-time progress metrics (from WebSocket updates)
  current_batch?: number;
  total_batches?: number;
  samples_processed?: number;
  total_samples?: number;
  samples_per_second?: number;
  eta_seconds?: number;
  features_in_heap?: number;
  heap_examples_count?: number;
  status_message?: string;  // Status message (e.g., "Saving features to database...")
  // Deletion progress (from WebSocket updates during background deletion)
  deletion_progress?: number;  // 0.0 to 1.0
  deletion_features_deleted?: number;
  deletion_total_features?: number;
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
 *
 * Supports both legacy (simple token list) and enhanced (context window) formats.
 * Enhanced format includes prefix/prime/suffix breakdown with positions.
 */
export interface FeatureActivationExample {
  // Legacy format (always present for backward compatibility)
  tokens: string[];
  activations: number[];
  max_activation: number;
  sample_index: number;

  // Enhanced context window format (optional, present in new extractions)
  prefix_tokens?: string[];
  prime_token?: string;
  suffix_tokens?: string[];
  prime_activation_index?: number;
  token_positions?: number[];
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
  // NLP Analysis (pre-computed, stored in database)
  nlp_analysis: NLPAnalysis | null;
  nlp_processed_at: string | null;
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
  sort_by?: 'activation_freq' | 'max_activation' | 'feature_id';
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
  // NLP Analysis fields are inherited from Feature
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

// ============================================
// NLP Analysis Types (Pre-computed feature analysis)
// ============================================

/**
 * Named entity from NLP analysis.
 */
export interface NLPNamedEntity {
  text: string;
  label: string;
  count: number;
}

/**
 * N-gram pattern from context analysis.
 */
export interface NLPNgram {
  tokens: string[];
  count: number;
}

/**
 * Syntactic pattern from context analysis.
 */
export interface NLPSyntacticPattern {
  pattern: string;
  count: number;
}

/**
 * High-activation token from activation analysis.
 */
export interface NLPHighActivationToken {
  token: string;
  activation: number;
}

/**
 * Semantic cluster from NLP analysis.
 */
export interface NLPSemanticCluster {
  label: string;
  example_indices: number[];
  size: number;
  representative_tokens: string[];
  avg_activation: number;
}

/**
 * Prime token analysis results.
 */
export interface NLPPrimeTokenAnalysis {
  unique_count: number;
  total_count: number;
  unique_tokens: string[];
  frequency_distribution: Record<string, number>;
  lowercase_distribution: Record<string, number>;
  pos_distribution: Record<string, number>;
  ner_entities: NLPNamedEntity[];
  token_types: Record<string, number>;
  most_common_token: [string, number];
  concentration_ratio: number;
}

/**
 * Context pattern analysis results.
 */
export interface NLPContextPatterns {
  prefix_bigrams: NLPNgram[];
  prefix_trigrams: NLPNgram[];
  suffix_bigrams: NLPNgram[];
  suffix_trigrams: NLPNgram[];
  immediately_before: Record<string, number>;
  immediately_after: Record<string, number>;
  syntactic_patterns: NLPSyntacticPattern[];
}

/**
 * Activation statistics from NLP analysis.
 */
export interface NLPActivationStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  skewness: number;
  distribution_type: 'symmetric' | 'right-skewed' | 'left-skewed' | 'unknown';
  high_activation_tokens: NLPHighActivationToken[];
  activation_range_buckets: Record<string, number>;
  coefficient_of_variation: number;
}

/**
 * Complete NLP analysis result stored on Feature.
 */
export interface NLPAnalysis {
  prime_token_analysis: NLPPrimeTokenAnalysis;
  context_patterns: NLPContextPatterns;
  activation_stats: NLPActivationStats;
  semantic_clusters: NLPSemanticCluster[];
  summary_for_prompt: string;
  num_examples_analyzed: number;
  computed_at: string;
}

/**
 * NLP analysis task status response.
 */
export interface NLPAnalysisTaskStatus {
  task_id: string;
  extraction_job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  message: string;
}

/**
 * NLP analysis progress event (from WebSocket).
 */
export interface NLPAnalysisProgressEvent {
  extraction_job_id: string;
  progress: number;
  features_analyzed: number;
  total_features: number;
  cached_count?: number;
  error_count?: number;
  status: 'analyzing' | 'completed' | 'failed';
  message: string;
  error?: string;
}
