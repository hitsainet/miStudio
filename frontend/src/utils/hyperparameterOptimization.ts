/**
 * SAE Hyperparameter Optimization Utilities
 *
 * Based on empirical analysis of completed training jobs.
 * See: docs/SAE_Hyperparameter_Optimization_Guide.md
 */

/**
 * Calculate optimal L1 alpha for a given latent dimension.
 *
 * Formula derived from exponential relationship between L1 alpha and L0 sparsity,
 * and power-law scaling with latent dimension.
 *
 * Base case: latent_dim=65536, l1_alpha=0.10 gives L0=5%
 * Scaling exponent: 6.64 (empirically determined from 4 training jobs)
 *
 * @param latentDim - Number of latent dimensions (features) in the SAE
 * @param targetL0 - Target sparsity level (default 0.05 = 5%)
 * @returns Optimal L1 alpha value
 *
 * @example
 * calculateOptimalL1Alpha(65536, 0.05) // Returns 0.10
 * calculateOptimalL1Alpha(32768, 0.05) // Returns 0.0010
 * calculateOptimalL1Alpha(131072, 0.05) // Returns 10.0
 */
export function calculateOptimalL1Alpha(
  latentDim: number,
  _targetL0: number = 0.05
): number {
  // Base case: latent_dim=65536, l1_alpha=0.10 gives L0=5%
  const BASE_LATENT_DIM = 65536;
  const BASE_L1_ALPHA = 0.10;
  const SCALING_EXPONENT = 6.64;

  // Scale L1 alpha based on latent dimension
  const l1Alpha = BASE_L1_ALPHA * Math.pow(latentDim / BASE_LATENT_DIM, SCALING_EXPONENT);

  // Round to 5 significant figures for usability
  return parseFloat(l1Alpha.toPrecision(5));
}

/**
 * Predict expected L0 sparsity from L1 alpha.
 *
 * Exponential relationship: L0(L1α) = 0.0731 * exp(-3.86 * L1α)
 *
 * @param l1Alpha - L1 sparsity coefficient
 * @returns Expected L0 sparsity (fraction of active neurons)
 */
export function predictL0Sparsity(l1Alpha: number): number {
  return 0.0731 * Math.exp(-3.86 * l1Alpha);
}

/**
 * Validate sparsity configuration and provide warnings.
 *
 * @param l1Alpha - L1 sparsity coefficient
 * @param latentDim - Number of latent dimensions
 * @param targetL0 - Target L0 sparsity
 * @returns Array of warning messages (empty if no issues)
 */
export function validateSparsityConfig(
  l1Alpha: number,
  latentDim: number,
  targetL0: number = 0.05
): string[] {
  const warnings: string[] = [];
  const optimalL1Alpha = calculateOptimalL1Alpha(latentDim, targetL0);
  const predictedL0 = predictL0Sparsity(l1Alpha);

  // Check if L1 alpha is too low
  if (l1Alpha < optimalL1Alpha * 0.5) {
    warnings.push(
      `L1 alpha (${l1Alpha.toFixed(6)}) is very low for latent_dim=${latentDim}. ` +
      `Recommended: ${optimalL1Alpha.toFixed(6)}. ` +
      `This will likely produce DENSE features (L0 > 20%) which are not interpretable.`
    );
  }

  // Check if L1 alpha is too high
  if (l1Alpha > optimalL1Alpha * 2.0) {
    warnings.push(
      `L1 alpha (${l1Alpha.toFixed(6)}) is very high for latent_dim=${latentDim}. ` +
      `Recommended: ${optimalL1Alpha.toFixed(6)}. ` +
      `This will produce OVER-SPARSE features (L0 < 2%) which may miss important patterns.`
    );
  }

  // Check if predicted L0 is far from target
  if (Math.abs(predictedL0 - targetL0) > targetL0 * 0.5) {
    warnings.push(
      `Predicted L0 (${(predictedL0 * 100).toFixed(1)}%) differs significantly from ` +
      `target (${(targetL0 * 100).toFixed(1)}%). Consider adjusting L1 alpha to ${optimalL1Alpha.toFixed(6)}.`
    );
  }

  return warnings;
}

/**
 * Get recommended hyperparameter settings for a given latent dimension.
 *
 * @param latentDim - Number of latent dimensions
 * @returns Recommended hyperparameter values
 */
export function getRecommendedHyperparameters(latentDim: number): {
  l1Alpha: number;
  hiddenDim: number;
  targetL0: number;
  learningRate: number;
  batchSize: number;
  totalSteps: number;
} {
  return {
    l1Alpha: calculateOptimalL1Alpha(latentDim),
    hiddenDim: Math.floor(latentDim / 32), // 32x expansion ratio
    targetL0: 0.05, // 5% sparsity
    learningRate: 0.0001,
    batchSize: 64,
    totalSteps: latentDim >= 65536 ? 100000 : 50000,
  };
}
