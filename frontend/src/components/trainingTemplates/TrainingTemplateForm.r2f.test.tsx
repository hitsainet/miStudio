/**
 * A Templates-panel save of the planned 16K template keeps what the form does not
 * show (R2D-4, review round 2 lane R2-F, 2026-09-15).
 *
 * These tests render the REAL form over template 6a460fbe's stored hyperparameters
 * (production export, 2026-09-14; the same fixture as TrainingTemplateForm.r2d.test.tsx).
 * They assert the WHOLE payload with toStrictEqual, so a key that goes missing, a
 * key that gains a value, and a key sent as `undefined` all fail.
 *
 * The expected payload is written out literally, not built by spreading the
 * fixture. A spread would agree with the overlay by construction.
 * backend/tests/unit/test_training_template_update_r2f.py sends this same payload
 * through the route and asserts what is stored.
 *
 * Mutation controls, each red here, restored by bytes with sha256 verified (the
 * table is in .claude/context/sessions/review_sae_remediation_R2_F_2026-09-15.md):
 *   F1 overlay dropped: the form sends its own fields only
 *   F2 edited field not applied: the stored value wins
 *   F3 cleared field resurrected: `undefined` keeps the stored value
 *   F4 resampling fields not owned: a TopK save carries the stored values
 *   F5 `description.trim() || undefined`: a cleared description comes back
 *   F6 `extra_metadata` sent as `undefined` when empty: cleared metadata comes back
 *   F7 `[datasetId]`: the datasets after the first are dropped
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TrainingTemplateForm } from './TrainingTemplateForm';
import { SAEArchitectureType } from '../../types/training';

const HYPERPARAMETERS_6A460FBE = {
  seed: null, aux_k: null, top_k: null, l1_alpha: null, bandwidth: 0.01, target_l0: null,
  batch_size: 2048, hidden_dim: 2048, hook_types: ['residual'], latent_dim: 16384,
  total_steps: 50000, adam_epsilon: null, log_interval: 100, warmup_steps: 2000,
  weight_decay: 0.0, learning_rate: 0.00007, ste_bandwidth: null, aux_loss_alpha: null,
  grad_clip_norm: 1.0, sparsity_coeff: 0.001, top_k_sparsity: null, dataset_weights: null,
  training_layers: [11, 12, 13], holdout_fraction: 0.0, architecture_type: 'jumprelu',
  evaluate_ce_delta: true, initial_threshold: 0.5, normalize_decoder: true,
  resample_interval: 5000, checkpoint_interval: 2000, dead_neuron_threshold: 10000,
  normalize_activations: 'constant_norm_rescale', resample_dead_neurons: true,
  sparsity_warmup_steps: 10000,
};

function template6a460fbe(overrides: Record<string, unknown> = {}) {
  return {
    id: '6a460fbe-1314-4b2e-b724-de2e032f807b',
    name: 'LFM2.5-1.2B · L11-13 residual · JumpReLU 8x',
    encoder_type: SAEArchitectureType.JUMPRELU,
    model_id: 'm_f0271325',
    dataset_ids: [],
    is_favorite: true,
    hyperparameters: { ...HYPERPARAMETERS_6A460FBE },
    ...overrides,
  };
}

function renderForm(template: Record<string, unknown>) {
  const onSubmit = vi.fn().mockResolvedValue(undefined);
  render(<TrainingTemplateForm template={template as any} onSubmit={onSubmit} />);
  return onSubmit;
}

async function save(onSubmit: ReturnType<typeof vi.fn>) {
  fireEvent.click(screen.getByRole('button', { name: /Update Template/ }));
  await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
  return onSubmit.mock.calls[0][0];
}

const setField = (label: RegExp | string, value: string) =>
  fireEvent.change(screen.getByLabelText(label), { target: { value } });

const openAdvanced = () => fireEvent.click(screen.getByRole('button', { name: /Advanced Settings/ }));

describe('R2-F: a Templates-panel save of the 16K template', () => {
  it('sends the whole template, with only the edited field changed', async () => {
    const onSubmit = renderForm(template6a460fbe());
    setField(/Total Steps/, '150000');

    const payload = await save(onSubmit);

    expect(payload).toStrictEqual({
      name: 'LFM2.5-1.2B · L11-13 residual · JumpReLU 8x',
      description: null,
      model_id: 'm_f0271325',
      dataset_ids: [],
      encoder_type: SAEArchitectureType.JUMPRELU,
      is_favorite: true,
      extra_metadata: {},
      hyperparameters: {
        // The edit.
        total_steps: 150000,
        // Not shown by the form: kept exactly as stored.
        seed: null,
        aux_k: null,
        top_k: null,
        hook_types: ['residual'],
        adam_epsilon: null,
        ste_bandwidth: null,
        aux_loss_alpha: null,
        top_k_sparsity: null,
        dataset_weights: null,
        training_layers: [11, 12, 13],
        holdout_fraction: 0,
        evaluate_ce_delta: true,
        normalize_decoder: true,
        normalize_activations: 'constant_norm_rescale',
        sparsity_warmup_steps: 10000,
        // Shown by the form: round-tripped through its fields unchanged.
        hidden_dim: 2048,
        latent_dim: 16384,
        architecture_type: SAEArchitectureType.JUMPRELU,
        learning_rate: 0.00007,
        batch_size: 2048,
        warmup_steps: 2000,
        weight_decay: 0,
        grad_clip_norm: 1,
        checkpoint_interval: 2000,
        log_interval: 100,
        resample_dead_neurons: true,
        resample_interval: 5000,
        dead_neuron_threshold: 10000,
        sparsity_coeff: 0.001,
        initial_threshold: 0.5,
        bandwidth: 0.01,
        // Always sent by buildLoopBlock; the template predates the field.
        lr_decay_steps: 0,
        // `l1_alpha` and `target_l0` are shown and empty (stored null), so they are
        // left out, and the backend stores its schema default: null.
      },
    });
  });

  it('does not bring back a field the user cleared', async () => {
    const onSubmit = renderForm(template6a460fbe());
    openAdvanced();
    setField('Gradient Clip Norm', '');

    const hp = (await save(onSubmit)).hyperparameters;

    expect('grad_clip_norm' in hp).toBe(false);
    expect(hp.training_layers).toEqual([11, 12, 13]);
  });

  it('switched to TopK, sends no resampling field and keeps the rest', async () => {
    const onSubmit = renderForm(template6a460fbe());
    fireEvent.click(screen.getByRole('radio', { name: /TopK/ }));

    const hp = (await save(onSubmit)).hyperparameters;

    for (const field of ['resample_dead_neurons', 'resample_interval', 'dead_neuron_threshold']) {
      expect(field in hp, field).toBe(false);
    }
    expect(hp).toMatchObject({
      architecture_type: SAEArchitectureType.TOPK,
      training_layers: [11, 12, 13],
      sparsity_warmup_steps: 10000,
      hook_types: ['residual'],
    });
  });

  it('clears a description and metadata the user emptied', async () => {
    const onSubmit = renderForm(
      template6a460fbe({ description: 'the old description', extra_metadata: { author: 'sean' } })
    );
    setField(/Description/, '');
    openAdvanced();
    setField(/Extra Metadata/, '{}');

    const payload = await save(onSubmit);

    expect(payload.description).toBeNull();
    expect(payload.extra_metadata).toStrictEqual({});
  });

  it('keeps the datasets the single dataset field does not show', async () => {
    const stored = { dataset_ids: ['ds_a', 'ds_b', 'ds_c'], dataset_id: 'ds_a' };

    const unchanged = renderForm(template6a460fbe(stored));
    expect((await save(unchanged)).dataset_ids).toEqual(['ds_a', 'ds_b', 'ds_c']);
  });

  it('replaces the first dataset when the field is edited', async () => {
    const stored = { dataset_ids: ['ds_a', 'ds_b', 'ds_c'], dataset_id: 'ds_a' };

    const edited = renderForm(template6a460fbe(stored));
    setField(/Dataset ID/, 'ds_z');
    expect((await save(edited)).dataset_ids).toEqual(['ds_z', 'ds_b', 'ds_c']);
  });

  it('drops only the first dataset when the field is cleared', async () => {
    const onSubmit = renderForm(template6a460fbe({ dataset_ids: ['ds_a', 'ds_b'], dataset_id: 'ds_a' }));
    setField(/Dataset ID/, '');
    expect((await save(onSubmit)).dataset_ids).toEqual(['ds_b']);
  });
});
