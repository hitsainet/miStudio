/**
 * The hyperparameters of a training request, built in ONE place for both payloads
 * the training panel sends: Start Training and Save as Template.
 *
 * WHY (review round 3, R3-C: R2D-6 and R2F-6, 2026-09-15). Both payloads listed
 * their keys by hand, and each list drifted from the other and from the schema.
 * - Start Training never sent `evaluate_ce_delta`, `evaluation_token_budget`,
 *   `holdout_eval_tokens`, `holdout_eval_chunk_tokens` or `seed`. A template that
 *   carried them loaded them into the form and lost them at launch, so a
 *   template's seed could not reproduce a run.
 * - Save as Template omitted `hook_types`, `log_interval`, `holdout_fraction`,
 *   `dataset_weights`, `seed` and `evaluate_ce_delta`. The backend stored its
 *   schema default for each, so a run training `['residual', 'mlp']` saved as
 *   `['residual']`, and a 0.1 held-out fraction saved as 0.
 *
 * THE ONE DIFFERENCE is what `dataset_weights` is positional over. A training's
 * weights follow the ids it trains on: `extraction_ids` on cached activations, in
 * the order they were PICKED, and `dataset_ids` on the fly. A template has no
 * extraction ids, so its weights follow its `dataset_ids`. The caller names the
 * reference; every other key comes from the same function.
 */

import type { TrainingConfig } from '../stores/trainingsStore';
import type { HyperparametersConfig } from '../types/training';
import type { TrainingTemplate } from '../types/trainingTemplate';
import { getFrameworkConfig } from '../config/frameworkConfigs';
import { buildLoopBlock } from './trainingLoopFields';
import { buildMixtureBlock } from './trainingMixture';

/** Backend bounds (`TrainingHyperparameters`). */
export const EVALUATION_TOKEN_BUDGET_MAX = 16_777_216;
export const HOLDOUT_EVAL_TOKENS_MAX = 10_000_000;
export const HOLDOUT_EVAL_CHUNK_TOKENS_MAX = 1_000_000;

/** Backend defaults, applied server-side when a field is omitted. */
export const DEFAULT_EVALUATION_TOKEN_BUDGET = 131_072;
export const DEFAULT_HOLDOUT_EVAL_TOKENS = 100_000;
export const DEFAULT_HOLDOUT_EVAL_CHUNK_TOKENS = 2_048;
export const DEFAULT_LOG_INTERVAL = 100;

/** The post-run evaluation and reproducibility fields of a request. */
export interface EvaluationBlock {
  evaluate_ce_delta?: boolean;
  evaluation_token_budget?: number;
  holdout_eval_tokens?: number;
  holdout_eval_chunk_tokens?: number;
  seed?: number;
}

type EvaluationInput = Pick<
  TrainingConfig,
  'evaluate_ce_delta' | 'evaluation_token_budget' | 'holdout_eval_tokens' | 'holdout_eval_chunk_tokens' | 'seed'
>;

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

/** A weight the backend accepts: finite and non-negative. */
const isUsableWeight = (value: unknown): value is number => isFiniteNumber(value) && value >= 0;

/**
 * The evaluation fields that are set. An unset field is omitted, never sent as a
 * guess, so the backend applies its own default: evaluation on, 131,072 tokens,
 * 100,000 held-out tokens in chunks of 2,048, and a seed derived from the training id.
 */
export function buildEvaluationBlock(input: EvaluationInput): EvaluationBlock {
  const block: EvaluationBlock = {};
  if (typeof input.evaluate_ce_delta === 'boolean') block.evaluate_ce_delta = input.evaluate_ce_delta;
  if (isFiniteNumber(input.evaluation_token_budget)) block.evaluation_token_budget = input.evaluation_token_budget;
  if (isFiniteNumber(input.holdout_eval_tokens)) block.holdout_eval_tokens = input.holdout_eval_tokens;
  if (isFiniteNumber(input.holdout_eval_chunk_tokens)) {
    block.holdout_eval_chunk_tokens = input.holdout_eval_chunk_tokens;
  }
  if (isFiniteNumber(input.seed)) block.seed = input.seed;
  return block;
}

function wholeNumberError(label: string, value: unknown, min: number, max: number): string | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== 'number' || !Number.isInteger(value) || value < min || value > max) {
    return `${label} must be a whole number from ${min.toLocaleString('en-US')} to ${max.toLocaleString('en-US')}`;
  }
  return null;
}

/**
 * The backend's bounds on the evaluation fields, checked before a request is sent.
 * Null when every set field is valid. A value the builder cannot send (NaN, a
 * fraction) is refused here rather than dropped, so it cannot silently become the
 * backend default.
 */
export function evaluationFieldsError(input: EvaluationInput): string | null {
  return (
    wholeNumberError('Evaluation token budget', input.evaluation_token_budget, 0, EVALUATION_TOKEN_BUDGET_MAX) ??
    wholeNumberError('Held-out tokens per log step', input.holdout_eval_tokens, 1, HOLDOUT_EVAL_TOKENS_MAX) ??
    wholeNumberError('Held-out tokens per chunk', input.holdout_eval_chunk_tokens, 1, HOLDOUT_EVAL_CHUNK_TOKENS_MAX) ??
    wholeNumberError('Seed', input.seed, 0, Number.MAX_SAFE_INTEGER)
  );
}

/** What `dataset_weights` is positional over, and the weight of each of those ids. */
export interface MixtureReference {
  /** The ids, in the order they are sent. */
  sourceIds: string[] | undefined;
  /** Weight per id; a missing entry weighs 1. */
  weightsBySource: Record<string, number> | undefined;
}

/**
 * The `hyperparameters` of a Start Training or a Save as Template request.
 *
 * Every key the form holds is sent from here. The sparsity-type blocks, the loop
 * block and the mixture block keep the rules their own modules document.
 */
export function buildTrainingHyperparameters(
  config: TrainingConfig,
  mixture: MixtureReference
): HyperparametersConfig {
  const sparsityType = getFrameworkConfig(config.architecture_type).sparsityType;
  return {
    hidden_dim: config.hidden_dim,
    latent_dim: config.latent_dim,
    architecture_type: config.architecture_type,
    training_layers: config.training_layers || [0],
    hook_types: config.hook_types || ['residual'],
    normalize_activations: config.normalize_activations,
    learning_rate: config.learning_rate,
    batch_size: config.batch_size,
    total_steps: config.total_steps,
    warmup_steps: config.warmup_steps,
    sparsity_warmup_steps: config.sparsity_warmup_steps,
    weight_decay: config.weight_decay,
    grad_clip_norm: config.grad_clip_norm,
    checkpoint_interval: config.checkpoint_interval,
    log_interval: config.log_interval,
    // LR decay, and the resampling fields for every framework that resamples.
    ...buildLoopBlock(config),
    // L1 frameworks
    ...(sparsityType === 'l1' && {
      l1_alpha: config.l1_alpha,
      target_l0: config.target_l0,
      normalize_decoder: config.normalize_decoder,
    }),
    // JumpReLU. target_l0 sets every threshold at step 0 by quantile calibration.
    ...(sparsityType === 'l0' && {
      sparsity_coeff: config.sparsity_coeff,
      target_l0: config.target_l0,
      initial_threshold: config.initial_threshold,
      bandwidth: config.bandwidth,
      ste_bandwidth: config.ste_bandwidth,
      normalize_decoder: config.normalize_decoder,
    }),
    // TopK
    ...(sparsityType === 'topk' && {
      top_k: config.top_k,
      aux_k: config.aux_k,
      aux_loss_alpha: config.aux_loss_alpha,
      adam_epsilon: config.adam_epsilon,
    }),
    // Mixture and held-out fraction; weights positional over `mixture.sourceIds`.
    ...buildMixtureBlock({
      extractionIds: mixture.sourceIds,
      weightsByExtraction: mixture.weightsBySource,
      holdoutFraction: config.holdout_fraction,
    }),
    // Post-run evaluation, held-out evaluation sizes and the seed.
    ...buildEvaluationBlock(config),
  };
}

/** A source a weight can be set for: an extraction or a dataset, and its dataset. */
export interface WeightedSource {
  sourceId: string;
  datasetId?: string;
}

/**
 * The weight each source trains with, keyed by source id.
 *
 * Its own entry first, then its dataset's, then 1. A template's weights are keyed by
 * dataset, because a template has no extraction ids. Undefined when no weight has
 * been set at all.
 */
export function resolveSourceWeights(
  sources: readonly WeightedSource[],
  weights: Record<string, number> | undefined
): Record<string, number> | undefined {
  if (!weights) return undefined;
  const resolved: Record<string, number> = {};
  for (const source of sources) {
    const own = weights[source.sourceId];
    const byDataset = source.datasetId === undefined ? undefined : weights[source.datasetId];
    resolved[source.sourceId] = isUsableWeight(own) ? own : isUsableWeight(byDataset) ? byDataset : 1;
  }
  return resolved;
}

/**
 * Resolved source weights re-keyed by dataset, for a template, whose weights are
 * positional over its `dataset_ids`. A dataset with no source weighs 1.
 */
export function weightsByDataset(
  datasetIds: readonly string[] | undefined,
  sources: readonly WeightedSource[],
  resolved: Record<string, number> | undefined
): Record<string, number> | undefined {
  if (!resolved || !datasetIds) return undefined;
  const byDataset: Record<string, number> = {};
  for (const datasetId of datasetIds) {
    const source = sources.find((candidate) => candidate.datasetId === datasetId);
    byDataset[datasetId] = source ? resolved[source.sourceId] ?? 1 : 1;
  }
  return byDataset;
}

/** The form update a template load applies after its hyperparameters, and weights it could not pair. */
export interface TemplateExtras {
  update: Partial<TrainingConfig>;
  /** The template's `dataset_weights` when they do not pair one-to-one with its `dataset_ids`. */
  unpairedWeights: number[] | null;
}

/**
 * The fields a template load writes out, so a template that lacks one takes the
 * backend default instead of the value the form happened to hold.
 *
 * Loading spreads a template's hyperparameters over the form. A key the template
 * lacks therefore kept the form's value. That was harmless for a field the start
 * payload never sent. Now that the payload sends them, a seed from the previous
 * template would ride into the next run unseen. So these are written out, the
 * way `configUpdatesFromTemplate` writes out the loop fields.
 *
 * `dataset_weights` load keyed by the template's `dataset_ids`. The panel resolves
 * each selected source to its dataset's weight, so the pairing is by id and never
 * by position. Weights that cannot be paired are returned for the panel to show,
 * never applied by position.
 */
export function configExtrasFromTemplate(template: TrainingTemplate): TemplateExtras {
  const hp = template.hyperparameters;
  const weights = Array.isArray(hp.dataset_weights) && hp.dataset_weights.length > 0 ? hp.dataset_weights : null;
  const datasetIds = template.dataset_ids ?? [];
  const paired = weights !== null && weights.length === datasetIds.length;
  return {
    update: {
      hook_types: hp.hook_types && hp.hook_types.length > 0 ? [...hp.hook_types] : ['residual'],
      log_interval: hp.log_interval ?? DEFAULT_LOG_INTERVAL,
      holdout_fraction: hp.holdout_fraction ?? 0,
      evaluate_ce_delta: hp.evaluate_ce_delta ?? undefined,
      evaluation_token_budget: hp.evaluation_token_budget ?? undefined,
      holdout_eval_tokens: hp.holdout_eval_tokens ?? undefined,
      holdout_eval_chunk_tokens: hp.holdout_eval_chunk_tokens ?? undefined,
      seed: hp.seed ?? undefined,
      dataset_weights_by_extraction:
        weights !== null && paired
          ? Object.fromEntries(datasetIds.map((datasetId, index) => [datasetId, weights[index]]))
          : undefined,
    },
    unpairedWeights: weights !== null && !paired ? weights : null,
  };
}
