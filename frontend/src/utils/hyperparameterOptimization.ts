/**
 * SAE Hyperparameter Optimization Utilities
 *
 * Based on SAELens defaults, Anthropic research, and empirical analysis.
 *
 * Key references:
 * - SAELens: https://github.com/jbloomAus/SAELens
 * - Anthropic "Scaling Monosemanticity" paper
 * - Gemma Scope paper for JumpReLU
 */

/**
 * Calculate optimal L1 alpha for a given latent dimension.
 *
 * The L1 coefficient controls sparsity in standard SAEs. Larger latent dimensions
 * (more features) require LESS L1 per feature to achieve the same overall sparsity,
 * because the total L1 penalty is summed across all active features.
 *
 * Formula: L1_alpha = BASE * sqrt(BASE_DIM / latentDim)
 *
 * This square-root scaling is derived from:
 * 1. SAELens default of ~5e-4 for 16K features
 * 2. The observation that L1 penalty accumulates linearly with active features
 * 3. To maintain ~5% L0 sparsity, we need proportionally less L1 as features increase
 *
 * @param latentDim - Number of latent dimensions (features) in the SAE
 * @param targetL0 - Target sparsity level (default 0.05 = 5%, affects scaling)
 * @returns Optimal L1 alpha value
 *
 * @example
 * calculateOptimalL1Alpha(16384, 0.05)  // Returns ~5e-4 (SAELens baseline)
 * calculateOptimalL1Alpha(6144, 0.05)   // Returns ~8.2e-4 (smaller SAE needs more L1)
 * calculateOptimalL1Alpha(65536, 0.05)  // Returns ~2.5e-4 (larger SAE needs less L1)
 * calculateOptimalL1Alpha(131072, 0.05) // Returns ~1.8e-4 (even larger)
 */
export function calculateOptimalL1Alpha(
  latentDim: number,
  targetL0: number = 0.05
): number {
  // Base case: SAELens default of ~5e-4 for 16K latent dimensions at 5% L0 target
  const BASE_LATENT_DIM = 16384;
  const BASE_L1_ALPHA = 5e-4;

  // Square-root scaling: larger SAEs need less L1, smaller SAEs need more
  // This maintains approximately constant L0 sparsity across different sizes
  let l1Alpha = BASE_L1_ALPHA * Math.sqrt(BASE_LATENT_DIM / latentDim);

  // Adjust for target L0: lower target sparsity (fewer active features) needs higher L1
  // Linear scaling around the 5% baseline
  const l0Adjustment = 0.05 / targetL0;
  l1Alpha *= l0Adjustment;

  // Clamp to reasonable range [1e-5, 1e-2]
  l1Alpha = Math.max(1e-5, Math.min(1e-2, l1Alpha));

  // Round to 2 significant figures for usability
  return parseFloat(l1Alpha.toPrecision(2));
}

/**
 * Predict expected L0 sparsity from L1 alpha and latent dimension.
 *
 * This is an approximation based on the inverse of calculateOptimalL1Alpha.
 * At the optimal L1 for a given latent_dim, we expect ~5% L0 sparsity.
 *
 * @param l1Alpha - L1 sparsity coefficient
 * @param latentDim - Number of latent dimensions (needed for scaling)
 * @returns Expected L0 sparsity (fraction of active neurons)
 */
export function predictL0Sparsity(l1Alpha: number, latentDim: number = 16384): number {
  // Get the optimal L1 for this latent dimension at 5% target
  const optimalL1 = calculateOptimalL1Alpha(latentDim, 0.05);

  // L0 scales roughly inversely with L1 alpha ratio
  // If L1 is at optimal, L0 ≈ 5%
  // If L1 is 2x optimal, L0 ≈ 2.5%
  // If L1 is 0.5x optimal, L0 ≈ 10%
  const ratio = optimalL1 / l1Alpha;
  const predictedL0 = 0.05 * ratio;

  // Clamp to reasonable range [0.001, 0.5]
  return Math.max(0.001, Math.min(0.5, predictedL0));
}

/**
 * Validate sparsity configuration and provide warnings.
 *
 * NOTE: This function is only relevant for Standard SAEs using L1 regularization.
 * JumpReLU SAEs use sparsity_coeff (L0 penalty) instead and should not show
 * these warnings.
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

  // Format L1 values nicely (scientific notation for small values)
  const formatL1 = (val: number) => {
    if (val < 0.001) return val.toExponential(1);
    return val.toFixed(4);
  };

  // Check if L1 alpha is too low (will produce dense, uninterpretable features)
  if (l1Alpha < optimalL1Alpha * 0.3) {
    warnings.push(
      `L1 alpha (${formatL1(l1Alpha)}) is very low for latent_dim=${latentDim.toLocaleString()}. ` +
      `Recommended: ${formatL1(optimalL1Alpha)}. ` +
      `This will likely produce DENSE features (L0 > 15%) which are harder to interpret.`
    );
  }

  // Check if L1 alpha is too high (will produce over-sparse features)
  if (l1Alpha > optimalL1Alpha * 3.0) {
    warnings.push(
      `L1 alpha (${formatL1(l1Alpha)}) is very high for latent_dim=${latentDim.toLocaleString()}. ` +
      `Recommended: ${formatL1(optimalL1Alpha)}. ` +
      `This will produce OVER-SPARSE features (L0 < 2%) which may miss important patterns.`
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
