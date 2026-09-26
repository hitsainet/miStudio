/**
 * The one builder behind Start Training and Save as Template (review round 3, R3-C).
 *
 * R2D-6: the start payload never sent `evaluate_ce_delta`, `evaluation_token_budget`,
 * `holdout_eval_tokens`, `holdout_eval_chunk_tokens` or `seed`, even when a loaded
 * template carried them. R2F-6: Save as Template listed its keys by hand and dropped
 * `hook_types`, `log_interval`, `holdout_fraction`, `dataset_weights`, `seed` and
 * `evaluate_ce_delta`. The panel tests in `TrainingPanel.r3c.test.tsx` pin that both
 * requests go through this builder. This file pins what it builds.
 *
 * Every expected payload below is written out literally, never spread from the input,
 * so a key the builder drops or invents is a difference.
 *
 * MUTATION CONTROLS (the table, with results, is in
 * .claude/context/sessions/review_sae_remediation_R3_C_2026-09-15.md):
 *   C1  the evaluation block is not spread into the hyperparameters
 *   C2  `evaluate_ce_delta` sent only when true (a truthiness check)
 *   C3  a seed of 0 treated as unset (a truthiness check)
 *   C4  a template's dataset-keyed weight not used for a selected extraction
 *   C5  a missing seed in a template leaves the key out of the update (the form keeps its value)
 *   C6  unpaired template weights applied by position
 *   C7  the evaluation token budget's upper bound not checked
 */

import { describe, it, expect } from 'vitest';
import type { TrainingConfig } from '../stores/trainingsStore';
import { SAEArchitectureType } from '../types/training';
import type { TrainingTemplate } from '../types/trainingTemplate';
import {
  buildEvaluationBlock,
  buildTrainingHyperparameters,
  configExtrasFromTemplate,
  DEFAULT_HOLDOUT_FRACTION,
  evaluationFieldsError,
  resolveSourceWeights,
  weightsByDataset,
} from './trainingHyperparameters';

const jumpReluConfig: TrainingConfig = {
  model_id: 'm_lfm',
  dataset_ids: ['ds_code', 'ds_web'],
  hidden_dim: 2048,
  latent_dim: 16384,
  architecture_type: SAEArchitectureType.JUMPRELU,
  training_layers: [11, 12, 13],
  hook_types: ['residual', 'mlp'],
  l1_alpha: undefined,
  target_l0: 0.05,
  normalize_activations: 'constant_norm_rescale',
  holdout_fraction: 0.1,
  initial_threshold: 0.5,
  bandwidth: 0.01,
  ste_bandwidth: 0.5,
  sparsity_coeff: 1e-3,
  normalize_decoder: true,
  // Left over from a TopK selection: a JumpReLU request must not carry them.
  top_k: 64,
  aux_k: 128,
  learning_rate: 7e-5,
  batch_size: 2048,
  total_steps: 150000,
  warmup_steps: 2000,
  lr_decay_steps: 30000,
  sparsity_warmup_steps: 10000,
  weight_decay: 0,
  grad_clip_norm: 1,
  checkpoint_interval: 10000,
  log_interval: 250,
  dead_neuron_threshold: 10000,
  resample_dead_neurons: true,
  resample_interval: 5000,
  evaluate_ce_delta: false,
  evaluation_token_budget: 0,
  holdout_eval_tokens: 200000,
  holdout_eval_chunk_tokens: 4096,
  seed: 0,
};

describe('buildTrainingHyperparameters', () => {
  it('sends every key the form holds for a JumpReLU run, the evaluation fields and seed included', () => {
    const hp = buildTrainingHyperparameters(jumpReluConfig, {
      sourceIds: ['ds_web', 'ds_code'],
      weightsBySource: { ds_code: 0.25, ds_web: 0.75 },
    });

    expect(hp).toStrictEqual({
      hidden_dim: 2048,
      latent_dim: 16384,
      architecture_type: 'jumprelu',
      training_layers: [11, 12, 13],
      hook_types: ['residual', 'mlp'],
      normalize_activations: 'constant_norm_rescale',
      learning_rate: 7e-5,
      batch_size: 2048,
      total_steps: 150000,
      warmup_steps: 2000,
      sparsity_warmup_steps: 10000,
      weight_decay: 0,
      grad_clip_norm: 1,
      checkpoint_interval: 10000,
      log_interval: 250,
      lr_decay_steps: 30000,
      resample_dead_neurons: true,
      resample_interval: 5000,
      dead_neuron_threshold: 10000,
      sparsity_coeff: 1e-3,
      target_l0: 0.05,
      initial_threshold: 0.5,
      bandwidth: 0.01,
      ste_bandwidth: 0.5,
      normalize_decoder: true,
      // Positional over the sourceIds given ([web, code]), not the map's key order.
      dataset_weights: [0.75, 0.25],
      holdout_fraction: 0.1,
      // 0 and false are values, not "unset": each is sent.
      evaluate_ce_delta: false,
      evaluation_token_budget: 0,
      holdout_eval_tokens: 200000,
      holdout_eval_chunk_tokens: 4096,
      seed: 0,
    });
  });

  it('sends the TopK keys, no resampling or penalty keys, and no evaluation key the form never set', () => {
    const topk: TrainingConfig = {
      model_id: 'm_lfm',
      dataset_ids: ['ds_code'],
      hidden_dim: 2048,
      latent_dim: 16384,
      architecture_type: SAEArchitectureType.TOPK,
      training_layers: [12],
      hook_types: ['residual'],
      normalize_activations: 'none',
      // Stale values from an L1 or JumpReLU selection.
      l1_alpha: 5e-4,
      sparsity_coeff: 1e-3,
      target_l0: 0.05,
      resample_dead_neurons: true,
      top_k: 32,
      aux_k: 128,
      aux_loss_alpha: 1 / 32,
      adam_epsilon: 6.25e-10,
      learning_rate: 3e-4,
      batch_size: 4096,
      total_steps: 30000,
      warmup_steps: 1000,
      lr_decay_steps: 0,
      sparsity_warmup_steps: 0,
      weight_decay: 0,
      grad_clip_norm: 1,
      checkpoint_interval: 5000,
      log_interval: 100,
      holdout_fraction: 0,
    };

    expect(buildTrainingHyperparameters(topk, { sourceIds: ['ds_code'], weightsBySource: undefined })).toStrictEqual({
      hidden_dim: 2048,
      latent_dim: 16384,
      architecture_type: 'topk',
      training_layers: [12],
      hook_types: ['residual'],
      normalize_activations: 'none',
      learning_rate: 3e-4,
      batch_size: 4096,
      total_steps: 30000,
      warmup_steps: 1000,
      sparsity_warmup_steps: 0,
      weight_decay: 0,
      grad_clip_norm: 1,
      checkpoint_interval: 5000,
      log_interval: 100,
      lr_decay_steps: 0,
      top_k: 32,
      aux_k: 128,
      aux_loss_alpha: 1 / 32,
      adam_epsilon: 6.25e-10,
      // The form set 0, and 0 is SENT: omitted, it would be replaced by the
      // backend's 0.02 default.
      holdout_fraction: 0,
    });
  });
});

describe('buildEvaluationBlock', () => {
  it('omits every field that is unset, so the backend applies its default', () => {
    expect(buildEvaluationBlock({})).toStrictEqual({});
  });

  it('sends false, 0 and a seed of 0 as values', () => {
    expect(
      buildEvaluationBlock({ evaluate_ce_delta: false, evaluation_token_budget: 0, seed: 0 })
    ).toStrictEqual({ evaluate_ce_delta: false, evaluation_token_budget: 0, seed: 0 });
  });

  it('sends evaluation switched on explicitly', () => {
    expect(buildEvaluationBlock({ evaluate_ce_delta: true, seed: 1004206375 })).toStrictEqual({
      evaluate_ce_delta: true,
      seed: 1004206375,
    });
  });
});

describe('evaluationFieldsError', () => {
  it('accepts unset fields and every bound, inclusive', () => {
    expect(evaluationFieldsError({})).toBeNull();
    expect(
      evaluationFieldsError({
        evaluation_token_budget: 16_777_216,
        holdout_eval_tokens: 10_000_000,
        holdout_eval_chunk_tokens: 1_000_000,
        seed: 0,
      })
    ).toBeNull();
    expect(evaluationFieldsError({ evaluation_token_budget: 0, holdout_eval_tokens: 1, holdout_eval_chunk_tokens: 1 })).toBeNull();
  });

  it.each([
    [{ evaluation_token_budget: 16_777_217 }, 'Evaluation token budget must be a whole number from 0 to 16,777,216'],
    [{ evaluation_token_budget: -1 }, 'Evaluation token budget must be a whole number from 0 to 16,777,216'],
    [{ evaluation_token_budget: 1.5 }, 'Evaluation token budget must be a whole number from 0 to 16,777,216'],
    [{ holdout_eval_tokens: 0 }, 'Held-out tokens per log step must be a whole number from 1 to 10,000,000'],
    [{ holdout_eval_chunk_tokens: 1_000_001 }, 'Held-out tokens per chunk must be a whole number from 1 to 1,000,000'],
    [{ seed: -1 }, 'Seed must be a whole number from 0 to 9,007,199,254,740,991'],
    [{ seed: Number.NaN }, 'Seed must be a whole number from 0 to 9,007,199,254,740,991'],
  ])('refuses %j', (input, message) => {
    expect(evaluationFieldsError(input)).toBe(message);
  });
});

describe('source weights', () => {
  const cachedSources = [
    // Picked in the opposite order to dataset_ids.
    { sourceId: 'ext_web', datasetId: 'ds_web' },
    { sourceId: 'ext_code', datasetId: 'ds_code' },
  ];

  it('is undefined when no weight was ever set', () => {
    expect(resolveSourceWeights(cachedSources, undefined)).toBeUndefined();
  });

  it("uses a source's own weight, then its dataset's, then 1", () => {
    expect(resolveSourceWeights(cachedSources, { ds_code: 3, ds_web: 1 })).toStrictEqual({ ext_web: 1, ext_code: 3 });
    expect(resolveSourceWeights(cachedSources, { ext_web: 2, ds_web: 9, ds_code: 3 })).toStrictEqual({
      ext_web: 2,
      ext_code: 3,
    });
    expect(resolveSourceWeights(cachedSources, { ds_other: 5 })).toStrictEqual({ ext_web: 1, ext_code: 1 });
  });

  it('re-keys resolved weights by dataset, in dataset order, for a template', () => {
    const resolved = { ext_web: 0.75, ext_code: 0.25 };
    expect(weightsByDataset(['ds_code', 'ds_web', 'ds_orphan'], cachedSources, resolved)).toStrictEqual({
      ds_code: 0.25,
      ds_web: 0.75,
      ds_orphan: 1,
    });
    expect(weightsByDataset(['ds_code'], cachedSources, undefined)).toBeUndefined();
  });
});

const template = (hyperparameters: Record<string, unknown>, datasetIds: string[]) =>
  ({
    id: 'tmpl',
    name: 'tmpl',
    model_id: null,
    dataset_ids: datasetIds,
    encoder_type: SAEArchitectureType.JUMPRELU,
    is_favorite: false,
    hyperparameters: { hidden_dim: 2048, latent_dim: 16384, learning_rate: 7e-5, batch_size: 2048, total_steps: 50000, ...hyperparameters },
  }) as unknown as TrainingTemplate;

describe('configExtrasFromTemplate', () => {
  it("writes out a template's evaluation fields, seed and weights keyed by its datasets", () => {
    const extras = configExtrasFromTemplate(
      template(
        {
          hook_types: ['residual', 'attention'],
          log_interval: 250,
          holdout_fraction: 0.05,
          evaluate_ce_delta: false,
          evaluation_token_budget: 65536,
          holdout_eval_tokens: 50000,
          holdout_eval_chunk_tokens: 1024,
          seed: 7,
          dataset_weights: [0.2, 0.8],
        },
        ['ds_code', 'ds_web']
      )
    );
    expect(extras).toStrictEqual({
      update: {
        hook_types: ['residual', 'attention'],
        log_interval: 250,
        holdout_fraction: 0.05,
        evaluate_ce_delta: false,
        evaluation_token_budget: 65536,
        holdout_eval_tokens: 50000,
        holdout_eval_chunk_tokens: 1024,
        seed: 7,
        dataset_weights_by_extraction: { ds_code: 0.2, ds_web: 0.8 },
      },
      unpairedWeights: null,
    });
  });

  it('writes out the backend default for every field a template lacks or stores as null', () => {
    const extras = configExtrasFromTemplate(
      template({ seed: null, dataset_weights: null, holdout_eval_tokens: null, evaluate_ce_delta: true }, ['ds_code'])
    );
    expect(extras).toStrictEqual({
      update: {
        hook_types: ['residual'],
        log_interval: 100,
        // The BACKEND default, which is now 0.02 — not 0. Sending an explicit 0
        // here would silently override it for every template predating the field.
        holdout_fraction: DEFAULT_HOLDOUT_FRACTION,
        evaluate_ce_delta: true,
        evaluation_token_budget: undefined,
        holdout_eval_tokens: undefined,
        holdout_eval_chunk_tokens: undefined,
        seed: undefined,
        dataset_weights_by_extraction: undefined,
      },
      unpairedWeights: null,
    });
    // Written out, so the store's merge replaces a previous template's seed.
    expect('seed' in extras.update).toBe(true);
    expect('dataset_weights_by_extraction' in extras.update).toBe(true);
  });

  it('keeps an explicit holdout of 0, so a historical run can be reproduced exactly', () => {
    const extras = configExtrasFromTemplate(template({ holdout_fraction: 0 }, ['ds_code']));
    expect(extras.update.holdout_fraction).toBe(0);
  });

  it('holds out 2% by default, matching the backend schema', () => {
    expect(DEFAULT_HOLDOUT_FRACTION).toBe(0.02);
  });

  it('returns weights it cannot pair with the datasets, and applies none of them by position', () => {
    // The stored 16K template 6a460fbe names no datasets.
    const extras = configExtrasFromTemplate(template({ dataset_weights: [1, 2, 3, 4, 5] }, []));
    expect(extras.update.dataset_weights_by_extraction).toBeUndefined();
    expect(extras.unpairedWeights).toStrictEqual([1, 2, 3, 4, 5]);
  });
});
