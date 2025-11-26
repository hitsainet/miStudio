/**
 * Helper functions for generating training template names and descriptions.
 */

import { SAEArchitectureType } from '../types/training';

/**
 * Generate a descriptive template name from configuration parameters.
 * Format: {ModelName}_{DatasetName}_{Architecture}_{Differentiator}
 *
 * @param modelName - Name of the model (e.g., "TinyLlama-1.1B")
 * @param datasetName - Name of the dataset (e.g., "OpenWebText")
 * @param architecture - SAE architecture type
 * @param hyperparameters - Training hyperparameters object
 * @returns Generated template name
 *
 * @example
 * generateTemplateName("TinyLlama-1.1B", "OpenWebText", "standard", { l1_alpha: 0.0001 })
 * // Returns: "TinyLlama_OpenWebText_Standard_L1-0.0001"
 */
export function generateTemplateName(
  modelName: string,
  datasetName: string,
  architecture: SAEArchitectureType,
  hyperparameters: {
    l1_alpha?: number;
    target_l0?: number;
    learning_rate?: number;
    latent_dim?: number;
    hidden_dim?: number;
  }
): string {
  // Clean up model name (remove special chars, limit length)
  const cleanModelName = modelName
    .replace(/[^a-zA-Z0-9-]/g, '')
    .split('-')[0]
    .slice(0, 20);

  // Clean up dataset name
  const cleanDatasetName = datasetName
    .replace(/[^a-zA-Z0-9-]/g, '')
    .slice(0, 20);

  // Format architecture name
  const archName = architecture === SAEArchitectureType.STANDARD
    ? 'Standard'
    : architecture === SAEArchitectureType.SKIP
    ? 'Skip'
    : 'Transcoder';

  // Determine differentiator based on key hyperparameters
  let differentiator: string;

  if (hyperparameters.l1_alpha !== undefined) {
    // Use L1 alpha as differentiator
    const l1Value = hyperparameters.l1_alpha;
    if (l1Value >= 0.001) {
      differentiator = 'HighSparsity';
    } else if (l1Value >= 0.0001) {
      differentiator = `L1-${l1Value.toFixed(4)}`;
    } else {
      differentiator = `L1-${l1Value.toExponential(1)}`;
    }
  } else if (hyperparameters.target_l0 !== undefined) {
    // Use target L0 as differentiator
    const l0Percent = (hyperparameters.target_l0 * 100).toFixed(0);
    differentiator = `L0-${l0Percent}pct`;
  } else if (hyperparameters.latent_dim && hyperparameters.hidden_dim) {
    // Use dictionary expansion as differentiator
    const multiplier = Math.round(hyperparameters.latent_dim / hyperparameters.hidden_dim);
    differentiator = `Dict${multiplier}x`;
  } else {
    // Default differentiator
    differentiator = 'Custom';
  }

  return `${cleanModelName}_${cleanDatasetName}_${archName}_${differentiator}`;
}

/**
 * Generate a descriptive summary of key hyperparameters.
 * Format: L1: {value} | LR: {value} | Dict: {hidden}→{latent} ({mult}x) | Steps: {steps} | Target L0: {percent}%
 *
 * @param hyperparameters - Training hyperparameters object
 * @returns Generated description string
 *
 * @example
 * generateTemplateDescription({ l1_alpha: 0.0001, learning_rate: 0.00027, hidden_dim: 2048, latent_dim: 8192, total_steps: 50000, target_l0: 0.05 })
 * // Returns: "L1: 0.0001 | LR: 0.00027 | Dict: 2048→8192 (4x) | Steps: 50k | Target L0: 5%"
 */
export function generateTemplateDescription(hyperparameters: {
  l1_alpha?: number;
  learning_rate?: number;
  hidden_dim?: number;
  latent_dim?: number;
  total_steps?: number;
  target_l0?: number;
  batch_size?: number;
  warmup_steps?: number;
  top_k_sparsity?: number;
}): string {
  const parts: string[] = [];

  // L1 Alpha
  if (hyperparameters.l1_alpha !== undefined) {
    const l1 = hyperparameters.l1_alpha;
    const l1Str = l1 >= 0.001 ? l1.toFixed(4) : l1.toExponential(2);
    parts.push(`L1: ${l1Str}`);
  }

  // Learning Rate
  if (hyperparameters.learning_rate !== undefined) {
    const lr = hyperparameters.learning_rate;
    const lrStr = lr >= 0.001 ? lr.toFixed(4) : lr.toExponential(2);
    parts.push(`LR: ${lrStr}`);
  }

  // Dictionary dimensions
  if (hyperparameters.hidden_dim && hyperparameters.latent_dim) {
    const multiplier = (hyperparameters.latent_dim / hyperparameters.hidden_dim).toFixed(0);
    parts.push(`Dict: ${hyperparameters.hidden_dim}→${hyperparameters.latent_dim} (${multiplier}x)`);
  }

  // Total steps (format with k/M suffix)
  if (hyperparameters.total_steps !== undefined) {
    const steps = hyperparameters.total_steps;
    let stepsStr: string;
    if (steps >= 1000000) {
      stepsStr = `${(steps / 1000000).toFixed(1)}M`;
    } else if (steps >= 1000) {
      stepsStr = `${(steps / 1000).toFixed(0)}k`;
    } else {
      stepsStr = steps.toString();
    }
    parts.push(`Steps: ${stepsStr}`);
  }

  // Target L0 sparsity
  if (hyperparameters.target_l0 !== undefined) {
    const l0Percent = (hyperparameters.target_l0 * 100).toFixed(0);
    parts.push(`Target L0: ${l0Percent}%`);
  }

  // Top-K sparsity (if specified)
  if (hyperparameters.top_k_sparsity !== undefined) {
    parts.push(`Top-K: ${hyperparameters.top_k_sparsity.toFixed(1)}%`);
  }

  // Batch size (if non-default)
  if (hyperparameters.batch_size !== undefined && hyperparameters.batch_size !== 128) {
    parts.push(`Batch: ${hyperparameters.batch_size}`);
  }

  // Warmup steps (if specified)
  if (hyperparameters.warmup_steps !== undefined && hyperparameters.warmup_steps > 0) {
    const warmupK = (hyperparameters.warmup_steps / 1000).toFixed(0);
    parts.push(`Warmup: ${warmupK}k`);
  }

  return parts.join(' | ');
}

/**
 * Validate template name for length and allowed characters.
 */
export function validateTemplateName(name: string): { valid: boolean; error?: string } {
  if (!name || name.trim().length === 0) {
    return { valid: false, error: 'Template name cannot be empty' };
  }

  if (name.length > 100) {
    return { valid: false, error: 'Template name must be 100 characters or less' };
  }

  // Allow alphanumeric, spaces, hyphens, underscores
  if (!/^[a-zA-Z0-9\s_-]+$/.test(name)) {
    return { valid: false, error: 'Template name can only contain letters, numbers, spaces, hyphens, and underscores' };
  }

  return { valid: true };
}
