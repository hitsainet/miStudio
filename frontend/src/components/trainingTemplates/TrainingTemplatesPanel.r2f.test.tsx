/**
 * The Templates panel's Edit and Duplicate paths send the whole template to the API
 * (R2D-4, review round 2 lane R2-F, 2026-09-15).
 *
 * The form tests mock `onSubmit`. These render the real panel, store and form, and
 * mock only the HTTP client module, so what is asserted is the request body the
 * browser would send: its payload and its call count.
 *
 * Duplicate matters as much as Edit. It opens the same form over a copy and CREATES
 * from it, and before the fix a duplicate of the 16K template came out with default
 * layers (the backend fills `training_layers` [0] on create).
 *
 * Mutation controls (red here; the table is in the R2-F record):
 *   F1 overlay dropped in the form (both tests)
 *   F8 the panel's duplicate copies no hyperparameters (`hyperparameters: {}`)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

vi.mock('../../api/trainingTemplates', () => ({
  getTrainingTemplates: vi.fn(),
  getFavoriteTrainingTemplates: vi.fn(),
  getTrainingTemplate: vi.fn(),
  createTrainingTemplate: vi.fn(),
  updateTrainingTemplate: vi.fn(),
  deleteTrainingTemplate: vi.fn(),
  toggleTrainingTemplateFavorite: vi.fn(),
  exportTrainingTemplates: vi.fn(),
  importTrainingTemplates: vi.fn(),
}));

import * as api from '../../api/trainingTemplates';
import { TrainingTemplatesPanel } from '../panels/TrainingTemplatesPanel';
import { useTrainingTemplatesStore } from '../../stores/trainingTemplatesStore';

const HYPERPARAMETERS = {
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

const TEMPLATE = {
  id: '6a460fbe-1314-4b2e-b724-de2e032f807b',
  name: 'LFM2.5-1.2B · L11-13 residual · JumpReLU 8x',
  encoder_type: 'jumprelu',
  model_id: 'm_f0271325',
  dataset_ids: [],
  dataset_id: null,
  is_favorite: true,
  extra_metadata: {},
  hyperparameters: HYPERPARAMETERS,
  created_at: '2026-09-14T00:00:00Z',
  updated_at: '2026-09-14T00:00:00Z',
};

/** Keys the form has no control for: each must reach the API unchanged. */
const UNEXPOSED = {
  training_layers: [11, 12, 13],
  hook_types: ['residual'],
  sparsity_warmup_steps: 10000,
  normalize_activations: 'constant_norm_rescale',
  normalize_decoder: true,
  evaluate_ce_delta: true,
  holdout_fraction: 0,
  seed: null,
  dataset_weights: null,
  ste_bandwidth: null,
};

beforeEach(() => {
  vi.mocked(api.getTrainingTemplates).mockReset().mockResolvedValue({
    data: [TEMPLATE as any],
    pagination: { page: 1, limit: 50, total: 1, total_pages: 1, has_next: false, has_prev: false },
  });
  vi.mocked(api.getFavoriteTrainingTemplates).mockReset().mockResolvedValue({ data: [] });
  vi.mocked(api.updateTrainingTemplate).mockReset().mockResolvedValue(TEMPLATE as any);
  vi.mocked(api.createTrainingTemplate)
    .mockReset()
    .mockResolvedValue({ ...TEMPLATE, id: 'copy-0000', name: `${TEMPLATE.name} (Copy)` } as any);
  useTrainingTemplatesStore.setState({ templates: [], favorites: [], loading: false, error: null });
});

async function openFromCard(title: 'Edit template' | 'Duplicate template') {
  render(<TrainingTemplatesPanel />);
  fireEvent.click(await screen.findByTitle(title));
  fireEvent.change(screen.getByLabelText(/Total Steps/), { target: { value: '150000' } });
  fireEvent.click(screen.getByRole('button', { name: /Update Template/ }));
}

describe('R2-F: the Templates panel sends the whole template', () => {
  it('Edit: one PATCH, for this template, carrying every key the form does not show', async () => {
    await openFromCard('Edit template');

    await waitFor(() => expect(api.updateTrainingTemplate).toHaveBeenCalledTimes(1));
    expect(api.createTrainingTemplate).not.toHaveBeenCalled();
    const [id, body] = vi.mocked(api.updateTrainingTemplate).mock.calls[0];
    expect(id).toBe(TEMPLATE.id);
    expect(body.hyperparameters).toMatchObject({ ...UNEXPOSED, total_steps: 150000 });
  });

  it('Duplicate: one POST, carrying every key the form does not show', async () => {
    await openFromCard('Duplicate template');

    await waitFor(() => expect(api.createTrainingTemplate).toHaveBeenCalledTimes(1));
    expect(api.updateTrainingTemplate).not.toHaveBeenCalled();
    const [body] = vi.mocked(api.createTrainingTemplate).mock.calls[0];
    expect(body.name).toBe(`${TEMPLATE.name} (Copy)`);
    expect(body.hyperparameters).toMatchObject({ ...UNEXPOSED, total_steps: 150000 });
  });
});
