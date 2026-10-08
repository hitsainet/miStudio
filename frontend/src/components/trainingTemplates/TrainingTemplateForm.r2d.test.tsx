/**
 * R2-D reproduction (review round 2 of the SAE training remediation, 2026-09-15). REPORT-ONLY.
 *
 * R2D-4: saving a template from the Templates panel deletes every hyperparameter the form
 * does not expose. The form rebuilds `hyperparameters` from its own fields, and
 * `TrainingTemplateService.update_template` stores `updates.model_dump(exclude_unset=True)`,
 * which recurses into the nested model: only the keys the form sent survive (checked on the
 * branch: a six-key update stores exactly six keys). Template 6a460fbe, the planned 16K run,
 * holds `training_layers` [11, 12, 13], `sparsity_warmup_steps` 10,000 and more that this
 * form never sends, so opening it to change `total_steps` and pressing Update Template
 * silently strips them. Loaded into the training form afterwards, the missing keys fall back
 * to whatever the form already held.
 *
 * The CONTROL proves the harness: the same template, the same submit, and a field the form
 * does expose arrives. The reproduction is `it.fails`: it passes while the defect stands and
 * turns red once the form (or the update) keeps the template's other keys. NEGATIVE CONTROL
 * WHEN FIXED: change `it.fails` to `it`, confirm green, revert the fix, confirm red.
 *
 * FIXED (review round 2, lane R2-F, 2026-09-15). The form now overlays its edits on
 * the template's stored hyperparameters (`templateFormPayload.ts`), and the update
 * stores the complete dump. `it.fails` became `it`: green with the fix. With
 * TrainingTemplateForm.tsx reverted to 25b3df86 it is red again, all six keys absent.
 * The whole-payload assertions are in TrainingTemplateForm.r2f.test.tsx.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TrainingTemplateForm } from './TrainingTemplateForm';
import { SAEArchitectureType } from '../../types/training';

/** Template 6a460fbe's stored hyperparameters (production export, 2026-09-14). */
const TEMPLATE_6A460FBE = {
  id: '6a460fbe-1314-4b2e-b724-de2e032f807b',
  name: 'LFM2.5-1.2B · L11-13 residual · JumpReLU 8x',
  encoder_type: SAEArchitectureType.JUMPRELU,
  model_id: 'm_f0271325',
  dataset_ids: [],
  is_favorite: true,
  hyperparameters: {
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
  },
};

async function saveTemplateWithTotalSteps(totalSteps: string) {
  const onSubmit = vi.fn().mockResolvedValue(undefined);
  render(<TrainingTemplateForm template={TEMPLATE_6A460FBE as any} onSubmit={onSubmit} />);
  fireEvent.change(screen.getByLabelText(/Total Steps/), { target: { value: totalSteps } });
  fireEvent.click(screen.getByRole('button', { name: /Update Template/ }));
  await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
  return onSubmit.mock.calls[0][0].hyperparameters;
}

describe('R2-D: a Templates-panel save of the 16K template', () => {
  it('CONTROL: the edited field arrives', async () => {
    const hp = await saveTemplateWithTotalSteps('150000');
    expect(hp).toMatchObject({ total_steps: 150000, latent_dim: 16384, architecture_type: 'jumprelu' });
  });

  it('R2D-4: keeps the hyperparameters the form does not expose', async () => {
    const hp = await saveTemplateWithTotalSteps('150000');
    expect(hp).toMatchObject({
      training_layers: [11, 12, 13],
      hook_types: ['residual'],
      sparsity_warmup_steps: 10000,
      normalize_activations: 'constant_norm_rescale',
      normalize_decoder: true,
      evaluate_ce_delta: true,
    });
  });
});
