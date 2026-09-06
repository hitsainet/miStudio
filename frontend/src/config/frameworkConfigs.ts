/**
 * SAE Training Framework Configuration Registry
 *
 * Per-framework configuration based on published papers.
 * Each framework has its own optimizer settings, loss formulation,
 * and recommended hyperparameters.
 *
 * Used by:
 * - TrainingPanel (dropdown options, conditional field visibility)
 * - TrainingStore (apply framework defaults when architecture changes)
 * - TrainingCard (display framework name)
 * - TrainingTemplateForm (framework-aware form)
 */

import { SAEArchitectureType } from '../types/training';

/** Sparsity enforcement mechanism */
export type SparsityType = 'l1' | 'l0' | 'topk';

/** Framework configuration */
export interface FrameworkConfig {
  /** Display name for UI dropdown */
  displayName: string;
  /** Paper reference */
  paper: string;
  /** Short description shown below the dropdown */
  description: string;
  /** Sparsity enforcement type */
  sparsityType: SparsityType;
  /** Default hyperparameters to apply when this framework is selected */
  defaults: {
    learning_rate: number;
    l1_alpha?: number;
    sparsity_coeff?: number;
    top_k?: number;
    aux_loss_alpha?: number;
    adam_epsilon?: number;
    normalize_decoder?: boolean;
    initial_threshold?: number;
    bandwidth?: number;
    normalize_activations?: string;
    sparsity_warmup_steps?: number;
    resample_dead_neurons?: boolean;
  };
  /** Fields to show in the training form for this framework */
  visibleFields: string[];
  /** Fields that should be hidden for this framework */
  hiddenFields: string[];
}

/**
 * Framework configurations keyed by SAEArchitectureType.
 *
 * Each entry defines the display info, defaults, and field visibility
 * for a specific training framework.
 */
export const FRAMEWORK_CONFIGS: Record<string, FrameworkConfig> = {
  [SAEArchitectureType.STANDARD_SAELENS]: {
    displayName: 'Standard (SAELens)',
    paper: 'Bricken et al. 2023',
    description: 'Standard SAE with L1 sparsity and constant_norm_rescale normalization.',
    sparsityType: 'l1',
    defaults: {
      learning_rate: 4e-4,
      l1_alpha: 5e-4,
      normalize_activations: 'constant_norm_rescale',
      normalize_decoder: true,
      sparsity_warmup_steps: 5000,
      resample_dead_neurons: true,
    },
    visibleFields: [
      'l1_alpha', 'target_l0', 'sparsity_warmup_steps',
      'normalize_decoder', 'resample_dead_neurons', 'resample_interval',
      'dead_neuron_threshold',
    ],
    hiddenFields: [
      'top_k', 'aux_k', 'aux_loss_alpha', 'adam_epsilon',
      'initial_threshold', 'bandwidth', 'sparsity_coeff',
    ],
  },

  [SAEArchitectureType.STANDARD_ANTHROPIC]: {
    displayName: 'Standard (Anthropic)',
    paper: 'Templeton et al. 2024',
    description: 'Standard SAE with Anthropic normalization (E[||x||²]=d_model). L1 coefficient ~5.',
    sparsityType: 'l1',
    defaults: {
      learning_rate: 4e-4,
      l1_alpha: 5.0,
      normalize_activations: 'anthropic_rescale',
      normalize_decoder: true,
      sparsity_warmup_steps: 5000,
      resample_dead_neurons: true,
    },
    visibleFields: [
      'l1_alpha', 'target_l0', 'sparsity_warmup_steps',
      'normalize_decoder', 'resample_dead_neurons', 'resample_interval',
      'dead_neuron_threshold',
    ],
    hiddenFields: [
      'top_k', 'aux_k', 'aux_loss_alpha', 'adam_epsilon',
      'initial_threshold', 'bandwidth', 'sparsity_coeff',
    ],
  },

  [SAEArchitectureType.JUMPRELU]: {
    displayName: 'JumpReLU (Gemma Scope)',
    paper: 'Rajamanoharan et al. 2024',
    description: 'Learnable per-feature thresholds with L0 penalty via straight-through estimator.',
    sparsityType: 'l0',
    defaults: {
      learning_rate: 7e-5,
      sparsity_coeff: 1e-3,
      initial_threshold: 0.5,
      bandwidth: 0.01,
      normalize_activations: 'constant_norm_rescale',
      normalize_decoder: true,
      sparsity_warmup_steps: 10000,
      resample_dead_neurons: false,
    },
    visibleFields: [
      'sparsity_coeff', 'initial_threshold', 'bandwidth',
      'normalize_decoder', 'sparsity_warmup_steps',
      'dead_neuron_threshold',
    ],
    hiddenFields: [
      'l1_alpha', 'target_l0', 'top_k', 'aux_k', 'aux_loss_alpha',
      'adam_epsilon', 'resample_dead_neurons', 'resample_interval',
    ],
  },

  [SAEArchitectureType.TOPK]: {
    displayName: 'TopK (OpenAI)',
    paper: 'Gao et al. 2024',
    description: 'Structural sparsity via TopK selection. No sparsity penalty — uses auxiliary dead feature loss.',
    sparsityType: 'topk',
    defaults: {
      learning_rate: 3e-4,
      top_k: 64,
      aux_loss_alpha: 1 / 32,
      adam_epsilon: 6.25e-10,
      normalize_activations: 'constant_norm_rescale',
      normalize_decoder: false,
      sparsity_warmup_steps: 0,
      resample_dead_neurons: false,
    },
    visibleFields: [
      'top_k', 'aux_k', 'aux_loss_alpha', 'adam_epsilon',
    ],
    hiddenFields: [
      'l1_alpha', 'target_l0', 'sparsity_coeff',
      'initial_threshold', 'bandwidth', 'normalize_decoder',
      'sparsity_warmup_steps', 'resample_dead_neurons', 'resample_interval',
      'dead_neuron_threshold',
    ],
  },

  [SAEArchitectureType.SKIP]: {
    displayName: 'Skip',
    paper: 'Community variant',
    description: 'Standard SAE with residual skip connection added to decoder output.',
    sparsityType: 'l1',
    defaults: {
      learning_rate: 4e-4,
      l1_alpha: 5e-4,
      normalize_activations: 'constant_norm_rescale',
      normalize_decoder: true,
      sparsity_warmup_steps: 5000,
      resample_dead_neurons: true,
    },
    visibleFields: [
      'l1_alpha', 'target_l0', 'sparsity_warmup_steps',
      'normalize_decoder', 'resample_dead_neurons', 'resample_interval',
      'dead_neuron_threshold',
    ],
    hiddenFields: [
      'top_k', 'aux_k', 'aux_loss_alpha', 'adam_epsilon',
      'initial_threshold', 'bandwidth', 'sparsity_coeff',
    ],
  },

  [SAEArchitectureType.TRANSCODER]: {
    displayName: 'Transcoder',
    paper: 'Dunefsky et al. 2024',
    description: 'Predicts MLP output from MLP input. Uses L1 sparsity on latent features.',
    sparsityType: 'l1',
    defaults: {
      learning_rate: 4e-4,
      l1_alpha: 5e-4,
      normalize_activations: 'constant_norm_rescale',
      normalize_decoder: true,
      sparsity_warmup_steps: 5000,
      resample_dead_neurons: true,
    },
    visibleFields: [
      'l1_alpha', 'target_l0', 'sparsity_warmup_steps',
      'normalize_decoder', 'resample_dead_neurons', 'resample_interval',
      'dead_neuron_threshold',
    ],
    hiddenFields: [
      'top_k', 'aux_k', 'aux_loss_alpha', 'adam_epsilon',
      'initial_threshold', 'bandwidth', 'sparsity_coeff',
    ],
  },
};

/**
 * Get framework config for an architecture type.
 * Falls back to standard_saelens for unknown types.
 */
export function getFrameworkConfig(architectureType: string): FrameworkConfig {
  // Normalize legacy 'standard' to 'standard_saelens'
  const normalized = architectureType === 'standard' ? 'standard_saelens' : architectureType;
  return FRAMEWORK_CONFIGS[normalized] || FRAMEWORK_CONFIGS[SAEArchitectureType.STANDARD_SAELENS];
}

/**
 * Get display name for an architecture type.
 * Useful for TrainingCard and template displays.
 */
export function getFrameworkDisplayName(architectureType: string): string {
  return getFrameworkConfig(architectureType).displayName;
}

/**
 * Check if a field should be visible for the given architecture type.
 */
export function isFieldVisible(architectureType: string, fieldName: string): boolean {
  const config = getFrameworkConfig(architectureType);
  return config.visibleFields.includes(fieldName);
}

/**
 * Get all framework options for a dropdown selector.
 * Excludes the backward-compat 'standard' alias.
 */
export function getFrameworkOptions(): { value: string; label: string; description: string }[] {
  return [
    SAEArchitectureType.STANDARD_SAELENS,
    SAEArchitectureType.STANDARD_ANTHROPIC,
    SAEArchitectureType.JUMPRELU,
    SAEArchitectureType.TOPK,
    SAEArchitectureType.SKIP,
    SAEArchitectureType.TRANSCODER,
  ].map((type) => ({
    value: type,
    label: FRAMEWORK_CONFIGS[type].displayName,
    description: FRAMEWORK_CONFIGS[type].description,
  }));
}
