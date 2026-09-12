/**
 * Memory Estimation Utilities
 *
 * Provides GPU memory estimation for SAE training configurations.
 * Mirrors backend logic in backend/src/utils/resource_estimation.py
 */

export interface MemoryEstimation {
  /** Total estimated memory in GB */
  total_gb: number;
  /** Per-layer memory in GB */
  per_layer_gb: number;
  /** Model parameters memory in GB */
  model_memory_gb: number;
  /** Optimizer state memory in GB */
  optimizer_memory_gb: number;
  /** Activations memory in GB */
  activations_memory_gb: number;
  /** Base system memory in GB */
  base_memory_gb: number;
  /** Maximum number of layers that fit in 6GB */
  max_layers_in_6gb: number;
  /** Whether configuration fits in 6GB */
  fits_in_6gb: boolean;
  /** Warning message if exceeds 6GB */
  warning?: string;
}

/**
 * Estimate GPU memory requirements for multi-layer SAE training.
 *
 * Based on backend resource_estimation.py:estimate_multilayer_training_memory()
 *
 * @param hiddenDim - Hidden dimension (input/output size)
 * @param latentDim - Latent dimension (SAE width)
 * @param batchSize - Training batch size
 * @param numLayers - Number of layers to train
 * @param dtypeBytes - Bytes per parameter (4 for FP32, 2 for FP16)
 * @param safetyFactor - Safety margin multiplier (default: 1.3)
 * @param baseMemoryGb - Base system memory overhead (default: 2.0 GB)
 * @returns Memory estimation breakdown
 */
export function estimateMultilayerTrainingMemory(
  hiddenDim: number,
  latentDim: number,
  batchSize: number,
  numLayers: number,
  dtypeBytes: number = 4,
  safetyFactor: number = 1.3,
  baseMemoryGb: number = 2.0
): MemoryEstimation {
  // SAE model parameters: encoder (hidden_dim * latent_dim) + decoder (latent_dim * hidden_dim) + bias
  const encoderParams = hiddenDim * latentDim;
  const decoderParams = latentDim * hiddenDim;
  const biasParams = latentDim + hiddenDim;
  const totalParams = encoderParams + decoderParams + biasParams;

  // Model memory (FP32 or FP16)
  const modelMemoryBytes = totalParams * dtypeBytes;

  // Optimizer memory (Adam: 2x parameters for momentum and variance)
  const optimizerMemoryBytes = totalParams * dtypeBytes * 2;

  // Activation memory (forward + backward passes)
  const activationsMemoryBytes = batchSize * (hiddenDim + latentDim) * dtypeBytes * 2;

  // Per-layer memory
  const perLayerBytes = modelMemoryBytes + optimizerMemoryBytes + activationsMemoryBytes;
  const perLayerGb = (perLayerBytes / (1024 ** 3)) * safetyFactor;

  // Total memory for all layers
  const totalLayersMemory = perLayerGb * numLayers;
  const totalGb = baseMemoryGb + totalLayersMemory;

  // Calculate components for breakdown
  const modelMemoryGb = ((modelMemoryBytes * numLayers) / (1024 ** 3)) * safetyFactor;
  const optimizerMemoryGb = ((optimizerMemoryBytes * numLayers) / (1024 ** 3)) * safetyFactor;
  const activationsMemoryGb = ((activationsMemoryBytes * numLayers) / (1024 ** 3)) * safetyFactor;

  // Calculate max layers that fit in 6GB
  const availableForLayers = 6.0 - baseMemoryGb;
  const maxLayersIn6gb = Math.max(1, Math.floor(availableForLayers / perLayerGb));

  // Check if configuration fits
  const fitsIn6gb = totalGb <= 6.0;

  // Generate warning message
  let warning: string | undefined;
  if (!fitsIn6gb) {
    warning = `Training requires ${totalGb.toFixed(1)} GB. Reduce to ≤${maxLayersIn6gb} layer${maxLayersIn6gb !== 1 ? 's' : ''} to fit in 6GB.`;
  }

  return {
    total_gb: totalGb,
    per_layer_gb: perLayerGb,
    model_memory_gb: modelMemoryGb,
    optimizer_memory_gb: optimizerMemoryGb,
    activations_memory_gb: activationsMemoryGb,
    base_memory_gb: baseMemoryGb,
    max_layers_in_6gb: maxLayersIn6gb,
    fits_in_6gb: fitsIn6gb,
    warning,
  };
}

/**
 * Format memory size in GB with appropriate precision.
 *
 * @param gb - Memory size in GB
 * @returns Formatted string (e.g., "2.3 GB")
 */
export function formatMemorySize(gb: number): string {
  if (gb < 0.1) {
    return `${(gb * 1024).toFixed(0)} MB`;
  }
  return `${gb.toFixed(1)} GB`;
}
