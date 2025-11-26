/**
 * Steering Types
 *
 * TypeScript interfaces for Feature Steering feature.
 * Matches backend Pydantic schemas in src/schemas/steering.py
 *
 * Backend API Contract:
 * - POST /api/v1/steering/compare - Generate steering comparison
 * - POST /api/v1/steering/sweep - Run strength sweep
 * - GET /api/v1/steering/experiments - List saved experiments
 * - POST /api/v1/steering/experiments - Save experiment
 * - GET /api/v1/steering/experiments/:id - Get experiment
 * - DELETE /api/v1/steering/experiments/:id - Delete experiment
 */

/**
 * Color options for selected features.
 * Max 4 features, each with unique color.
 */
export type FeatureColor = 'teal' | 'blue' | 'purple' | 'amber';

/**
 * CSS classes for feature colors.
 */
export const FEATURE_COLORS: Record<FeatureColor, {
  bg: string;
  border: string;
  text: string;
  light: string;
}> = {
  teal: {
    bg: 'bg-teal-500',
    border: 'border-teal-500',
    text: 'text-teal-400',
    light: 'bg-teal-500/10',
  },
  blue: {
    bg: 'bg-blue-500',
    border: 'border-blue-500',
    text: 'text-blue-400',
    light: 'bg-blue-500/10',
  },
  purple: {
    bg: 'bg-purple-500',
    border: 'border-purple-500',
    text: 'text-purple-400',
    light: 'bg-purple-500/10',
  },
  amber: {
    bg: 'bg-amber-500',
    border: 'border-amber-500',
    text: 'text-amber-400',
    light: 'bg-amber-500/10',
  },
};

/**
 * Available feature colors in order.
 */
export const FEATURE_COLOR_ORDER: FeatureColor[] = ['teal', 'blue', 'purple', 'amber'];

/**
 * A feature selected for steering.
 */
export interface SelectedFeature {
  feature_idx: number;
  layer: number;
  strength: number; // -100 to +300
  label: string | null;
  color: FeatureColor;
}

/**
 * Generation parameters for text generation.
 */
export interface GenerationParams {
  max_new_tokens: number;
  temperature: number;
  top_p: number;
  top_k: number;
  num_samples: number;
  seed?: number;
}

/**
 * Advanced generation parameters.
 */
export interface AdvancedGenerationParams {
  repetition_penalty: number;
  presence_penalty: number;
  frequency_penalty: number;
  do_sample: boolean;
  stop_sequences: string[];
}

/**
 * Default generation parameters.
 */
export const DEFAULT_GENERATION_PARAMS: GenerationParams = {
  max_new_tokens: 100,
  temperature: 0.7,
  top_p: 0.9,
  top_k: 50,
  num_samples: 1,
};

/**
 * Request to generate a steering comparison.
 */
export interface SteeringComparisonRequest {
  sae_id: string;
  model_id?: string;
  prompt: string;
  selected_features: SelectedFeature[];
  generation_params?: GenerationParams;
  advanced_params?: AdvancedGenerationParams;
  include_unsteered?: boolean;
  compute_metrics?: boolean;
}

/**
 * Request for strength sweep.
 */
export interface SteeringStrengthSweepRequest {
  sae_id: string;
  model_id?: string;
  prompt: string;
  feature_idx: number;
  layer: number;
  strength_values: number[];
  generation_params?: GenerationParams;
}

/**
 * Generation quality metrics.
 */
export interface GenerationMetrics {
  perplexity: number | null;
  coherence: number | null;
  behavioral_score: number | null;
  token_count: number;
  generation_time_ms: number;
}

/**
 * Single steered output.
 */
export interface SteeredOutput {
  text: string;
  feature_config: SelectedFeature;
  metrics: GenerationMetrics | null;
}

/**
 * Unsteered baseline output.
 */
export interface UnsteeredOutput {
  text: string;
  metrics: GenerationMetrics | null;
}

/**
 * Steering comparison response.
 */
export interface SteeringComparisonResponse {
  comparison_id: string;
  sae_id: string;
  model_id: string;
  prompt: string;
  unsteered: UnsteeredOutput | null;
  steered: SteeredOutput[];
  metrics_summary: Record<string, any> | null;
  total_time_ms: number;
  created_at: string;
}

/**
 * Single strength sweep result.
 */
export interface StrengthSweepResult {
  strength: number;
  text: string;
  metrics: GenerationMetrics | null;
}

/**
 * Strength sweep response.
 */
export interface StrengthSweepResponse {
  sweep_id: string;
  sae_id: string;
  model_id: string;
  prompt: string;
  feature_idx: number;
  layer: number;
  unsteered: UnsteeredOutput;
  results: StrengthSweepResult[];
  total_time_ms: number;
  created_at: string;
}

/**
 * Request to save a steering experiment.
 */
export interface SteeringExperimentSaveRequest {
  name: string;
  description?: string;
  comparison_id: string;
  tags?: string[];
}

/**
 * Saved steering experiment.
 */
export interface SteeringExperiment {
  id: string;
  name: string;
  description: string | null;
  sae_id: string;
  model_id: string;
  prompt: string;
  selected_features: SelectedFeature[];
  generation_params: GenerationParams;
  results: SteeringComparisonResponse;
  tags: string[];
  created_at: string;
  updated_at: string;
}

/**
 * Paginated list of experiments.
 */
export interface SteeringExperimentListResponse {
  data: SteeringExperiment[];
  pagination: {
    skip: number;
    limit: number;
    total: number;
    has_more: boolean;
  };
}

/**
 * Real-time steering progress update (WebSocket).
 */
export interface SteeringProgressUpdate {
  comparison_id: string;
  status: string;
  current_config: string | null;
  progress: number;
  message: string | null;
}

/**
 * Feature activation analysis.
 */
export interface FeatureActivationAnalysis {
  feature_idx: number;
  activation_count: number;
  mean_activation: number;
  max_activation: number;
  activated_tokens: string[];
}

/**
 * Steering effect analysis (side effects).
 */
export interface SteeringEffectAnalysis {
  target_feature_idx: number;
  target_feature_activation_change: number;
  side_effects: FeatureActivationAnalysis[];
}

/**
 * Warning thresholds for steering strength.
 */
export const STRENGTH_THRESHOLDS = {
  CAUTION_LOW: -50,
  CAUTION_HIGH: 150,
  EXTREME_LOW: -80,
  EXTREME_HIGH: 250,
};

/**
 * Get warning level for a strength value.
 */
export function getStrengthWarningLevel(strength: number): 'normal' | 'caution' | 'extreme' {
  if (strength <= STRENGTH_THRESHOLDS.EXTREME_LOW || strength >= STRENGTH_THRESHOLDS.EXTREME_HIGH) {
    return 'extreme';
  }
  if (strength <= STRENGTH_THRESHOLDS.CAUTION_LOW || strength >= STRENGTH_THRESHOLDS.CAUTION_HIGH) {
    return 'caution';
  }
  return 'normal';
}

/**
 * Calculate the multiplier from strength.
 * Formula: multiplier = 1 + strength/100
 */
export function strengthToMultiplier(strength: number): number {
  return 1 + strength / 100;
}
