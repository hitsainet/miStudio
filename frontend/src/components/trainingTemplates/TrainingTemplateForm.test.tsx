/**
 * The template form's payload: LR decay and the dead-latent resampling fields.
 *
 * Driven through the rendered form — the fields a user edits are the fields
 * `onSubmit` receives. The form had no test file at all.
 *
 * Mutation controls F3, F4, F5 and F7 turn this file red; the full table is in
 * src/utils/trainingLoopFields.test.ts.
 *
 * Review round 1 (R1-A, 2026-09-15): F13b `warmup_steps || 1000` in the load
 * effect -> red; F17r the payload without buildLoopBlock -> red (4 tests).
 * F13a, the same `||` in the useState initialiser alone, survives as an
 * equivalent mutant: the effect runs on mount and replaces that value.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TrainingTemplateForm } from './TrainingTemplateForm';
import { SAEArchitectureType } from '../../types/training';

function renderForm() {
  const onSubmit = vi.fn().mockResolvedValue(undefined);
  render(<TrainingTemplateForm onSubmit={onSubmit} />);
  fireEvent.change(screen.getByLabelText(/Template Name/), { target: { value: 'my template' } });
  return onSubmit;
}

const setNumber = (label: string, value: string) =>
  fireEvent.change(screen.getByLabelText(label), { target: { value } });

describe('TrainingTemplateForm payload', () => {
  it('sends LR decay and every resampling field for a JumpReLU template', async () => {
    const onSubmit = renderForm();
    fireEvent.click(screen.getByRole('radio', { name: /JumpReLU/ }));
    fireEvent.click(screen.getByRole('button', { name: /Advanced Settings/ }));
    setNumber('LR Decay Steps', '20000');
    setNumber('Resample Interval', '2500');
    setNumber('Dead Neuron Threshold', '7000');

    fireEvent.click(screen.getByRole('button', { name: /Create Template/ }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0].hyperparameters).toMatchObject({
      architecture_type: SAEArchitectureType.JUMPRELU,
      lr_decay_steps: 20000,
      resample_dead_neurons: true,
      resample_interval: 2500,
      dead_neuron_threshold: 7000,
    });
  });

  it('carries an unticked resample flag', async () => {
    const onSubmit = renderForm();
    fireEvent.click(screen.getByRole('radio', { name: /JumpReLU/ }));
    fireEvent.click(screen.getByRole('button', { name: /Advanced Settings/ }));
    fireEvent.click(screen.getByRole('checkbox', { name: /Resample dead neurons/ }));

    fireEvent.click(screen.getByRole('button', { name: /Create Template/ }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0].hyperparameters.resample_dead_neurons).toBe(false);
  });

  it('neither shows nor sends the resampling fields for TopK', async () => {
    const onSubmit = renderForm();
    fireEvent.click(screen.getByRole('radio', { name: /TopK/ }));
    fireEvent.click(screen.getByRole('button', { name: /Advanced Settings/ }));
    expect(screen.queryByLabelText('Resample Interval')).toBeNull();
    expect(screen.queryByLabelText('Dead Neuron Threshold')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: /Create Template/ }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    const hp = onSubmit.mock.calls[0][0].hyperparameters;
    expect(hp.lr_decay_steps).toBe(0);
    for (const field of ['resample_dead_neurons', 'resample_interval', 'dead_neuron_threshold']) {
      expect(field in hp, field).toBe(false);
    }
  });

  it('loads a template saved with no warmup as no warmup, and saves it back that way', async () => {
    // `warmup_steps || 1000` loaded 0 as 1,000: the schedule check then refused a
    // decay that filled the run exactly, and a save wrote the 1,000 back.
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const template = {
      id: 'tmpl_nowarmup',
      name: 'no warmup',
      encoder_type: SAEArchitectureType.STANDARD_SAELENS,
      is_favorite: false,
      hyperparameters: {
        hidden_dim: 768, latent_dim: 16384, learning_rate: 3e-4, batch_size: 4096,
        total_steps: 10000, warmup_steps: 0, lr_decay_steps: 10000,
      },
    };
    render(<TrainingTemplateForm template={template as any} onSubmit={onSubmit} />);

    fireEvent.click(screen.getByRole('button', { name: /Update Template/ }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0].hyperparameters).toMatchObject({ warmup_steps: 0, lr_decay_steps: 10000 });
  });

  it('refuses warmup plus decay beyond total steps', async () => {
    const onSubmit = renderForm();
    fireEvent.click(screen.getByRole('button', { name: /Advanced Settings/ }));
    setNumber('LR Decay Steps', '100000'); // default total 100,000 + default warmup 1,000

    fireEvent.click(screen.getByRole('button', { name: /Create Template/ }));

    expect(await screen.findByText(/exceed total steps/)).toBeTruthy();
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
