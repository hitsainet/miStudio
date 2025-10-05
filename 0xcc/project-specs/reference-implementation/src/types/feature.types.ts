/**
 * Feature Types
 *
 * Backend API Contract:
 * - POST /api/features/extract - Extract features from trained SAE (long-running job)
 * - GET /api/features?trainingId=:id - List features for training
 * - GET /api/features/:id - Get detailed feature info
 * - GET /api/features/:id/activations - Get max activating examples
 * - GET /api/features/:id/logit-lens - Get logit lens analysis
 * - GET /api/features/:id/correlations - Get correlated features
 * - GET /api/features/:id/ablation-analysis - Get ablation impact
 * - GET /api/features/:id/heatmap - Get activation heatmap data
 * - PATCH /api/features/:id - Update feature (name, description, favorite)
 * - POST /api/features/projection - Compute UMAP/t-SNE projection
 * - GET /api/features/correlation-matrix - Get correlation matrix
 *
 * Feature extraction is a post-training analysis step
 */

/**
 * Discovered feature from trained SAE
 */
export interface Feature {
  id: number;
  training_id?: string;
  name: string; // User-provided or auto-generated
  description?: string;
  activation: number; // Average activation frequency (0-1)
  interpretability: number; // Interpretability score (0-1)
  layer?: number; // Which transformer layer
  neuron_index?: number; // Index in SAE latent space
  is_favorite?: boolean;
}

/**
 * Status of feature extraction process
 *
 * Feature extraction workflow:
 * 1. Run dataset through trained SAE
 * 2. Collect activation statistics
 * 3. Find max activating examples
 * 4. Compute interpretability scores
 *
 * Backend: Long-running job, use job queue (Celery/BullMQ)
 * Status polling: GET /api/features/extract/status/:jobId
 */
export interface FeatureExtractionStatus {
  job_id: string;
  status: 'idle' | 'extracting' | 'completed' | 'error';
  progress: number; // 0-100
  error?: string;
}

/**
 * Feature extraction request configuration
 */
export interface FeatureExtractionRequest {
  training_id: string;
  eval_samples: number; // How many samples to evaluate
  top_k_examples: number; // How many max activating examples per feature
}

/**
 * Example text sample with activation information
 *
 * Used for "max activating examples" - texts where feature activates strongest
 * Backend computes during feature extraction
 */
export interface ActivationExample {
  text: string;
  tokens: string[]; // Tokenized text
  activations: number[]; // Per-token activation values
  max_activation: number; // Highest activation in this example
  source?: string; // Dataset sample ID or source
}

/**
 * Logit lens analysis for feature
 *
 * Shows which tokens the feature predicts/correlates with
 * Backend implementation:
 * - Take feature activation vector
 * - Project through model's output layer: logits = W_unembed @ feature_vector
 * - Compute softmax probabilities
 * - Return top-k tokens
 */
export interface LogitLens {
  feature_id: number;
  top_tokens: string[];
  probabilities: number[];
  interpretation: string; // Human-readable interpretation
}

/**
 * Feature correlation with other features
 *
 * Backend computes Pearson correlation between feature activation patterns
 */
export interface FeatureCorrelation {
  id: number;
  name: string;
  correlation: number; // -1 to 1
}

/**
 * Ablation analysis results
 *
 * Backend implementation:
 * - Baseline: Compute perplexity with all features
 * - Ablation: Zero out specific feature, recompute perplexity
 * - Delta = perplexity_ablated - perplexity_baseline
 * - Larger delta = more important feature
 * - Cache results (expensive to compute)
 */
export interface AblationAnalysis {
  perplexity_delta: number; // Increase in perplexity when ablated
  impact_score: number; // 0-1, relative importance
}

/**
 * Activation heatmap data for visualization
 *
 * Shows which features activate for which tokens
 */
export interface ActivationHeatmap {
  tokens: string[];
  activations: number[][]; // [tokens][features]
  features: number[]; // Feature IDs
}

/**
 * UMAP/t-SNE projection for dimensionality reduction
 */
export interface FeatureProjection {
  feature_id: number;
  x: number;
  y: number;
  z?: number; // Optional for 3D projection
}

/**
 * Feature projection request
 */
export interface FeatureProjectionRequest {
  training_id: string;
  algorithm: 'umap' | 'tsne';
  n_components: 2 | 3;
  n_samples?: number; // Subsample if too large
}

/**
 * Correlation matrix for all features in a training
 */
export interface FeatureCorrelationMatrix {
  features: number[]; // Feature IDs
  matrix: number[][]; // Correlation values [-1, 1]
}

/**
 * Feature query parameters for list endpoint
 */
export interface FeatureQueryParams {
  training_id: string;
  search?: string;
  sort_by?: 'id' | 'activation_freq' | 'interpretability';
  order?: 'asc' | 'desc';
  page?: number;
  limit?: number;
}

/**
 * Feature update request
 */
export interface FeatureUpdateRequest {
  name?: string;
  description?: string;
  is_favorite?: boolean;
}
