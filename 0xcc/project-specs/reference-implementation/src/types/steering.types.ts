/**
 * Steering Types
 *
 * Backend API Contract:
 * - POST /api/steering/generate - Generate steered output
 *   Body: { modelId, trainingId, prompt, features, interventionLayer, temperature, maxTokens }
 *   Response: { unsteeredOutput, steeredOutput, metrics }
 * - GET /api/steering/presets - List steering presets
 * - POST /api/steering/presets - Save steering preset
 * - DELETE /api/steering/presets/:id - Delete preset
 *
 * Steering applies feature vectors to model activations at specified layer
 * Coefficients typically range from -5.0 to +5.0
 */

/**
 * Model steering configuration
 */
export interface SteeringConfig {
  selected_features: Feature[];
  coefficients: Record<number, number>; // featureId -> coefficient
  intervention_layer: number; // Which transformer layer to intervene at
  temperature: number; // Sampling temperature (0.1-2.0)
}

/**
 * Feature with steering coefficient
 */
export interface SteeringFeature {
  id: number;
  coefficient: number; // -5.0 to +5.0
}

/**
 * Steered generation request
 *
 * Backend implementation:
 * 1. Load base model
 * 2. Load SAE encoder/decoder from training
 * 3. Tokenize prompt
 * 4. Forward pass until intervention layer:
 *    - Get activations at layer N
 *    - Pass through SAE encoder: latents = encoder(activations)
 *    - Apply steering: latents[feature_ids] += coefficients
 *    - Reconstruct: modified_activations = decoder(latents)
 *    - Replace original activations with modified ones
 *    - Continue forward pass with modified activations
 * 5. Generate unsteered output (for comparison)
 * 6. Generate steered output (with intervention)
 * 7. Compute comparison metrics
 */
export interface SteeringGenerateRequest {
  model_id: string;
  training_id: string;
  prompt: string;
  features: SteeringFeature[];
  intervention_layer: number;
  temperature?: number; // Default 0.7
  max_tokens?: number; // Default 100
}

/**
 * Steered generation response
 */
export interface SteeringGenerateResponse {
  unsteered_output: string;
  steered_output: string;
  metrics: ComparisonMetrics;
}

/**
 * Comparison metrics between unsteered and steered outputs
 *
 * Backend computation:
 * - KL Divergence: KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))
 *   where P = unsteered probs, Q = steered probs
 * - Perplexity Delta: perplexity = exp(cross_entropy)
 * - Semantic Similarity: Cosine similarity of sentence embeddings
 * - Word Overlap: Jaccard similarity = |intersection| / |union|
 */
export interface ComparisonMetrics {
  kl_divergence: number; // KL divergence between token distributions
  perplexity_delta: number; // Perplexity change
  semantic_similarity: number; // Cosine similarity (0-1)
  word_overlap: number; // Jaccard similarity (0-1)
}

/**
 * Steering preset for saving/loading configurations
 */
export interface SteeringPreset {
  id: string;
  name: string;
  description?: string;
  features: SteeringFeature[];
  intervention_layer: number;
  created_at: string; // ISO 8601
}

/**
 * Steering preset creation request
 */
export interface SteeringPresetCreateRequest {
  name: string;
  description?: string;
  features: SteeringFeature[];
  intervention_layer: number;
}

// Re-export Feature type for convenience
import type { Feature } from './feature.types';
