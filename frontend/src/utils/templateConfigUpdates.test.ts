/**
 * Loading a template through the REAL training store (review round 1, R1-A).
 *
 * `updateConfig` resets a framework's defaults whenever an update changes the
 * architecture, AFTER merging that update. A template spread into one update lost
 * the values the reset writes — `resample_dead_neurons` and the learning rate
 * among them — and a template older than the loop fields kept the form's own
 * decay, interval and threshold. The panel test pins that `handleTemplateLoad`
 * sends exactly these updates; this file pins what they do to the store.
 *
 * Mutation controls (R1-A, recorded in
 * .claude/context/sessions/review_sae_remediation_R1_A_2026-09-15.md):
 *   F10 one update ({...hp, architecture_type})       -> red here
 *   F11 missing lr_decay_steps not written out           -> red here
 *   F12 store default threshold back to 10000            -> red here
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { useTrainingsStore } from '../stores/trainingsStore';
import { SAEArchitectureType } from '../types/training';
import type { TrainingTemplate } from '../types/trainingTemplate';
import { configUpdatesFromTemplate } from './templateConfigUpdates';
import { DEFAULT_DEAD_NEURON_THRESHOLD, DEFAULT_RESAMPLE_INTERVAL } from './trainingLoopFields';

const template = (encoder: SAEArchitectureType, hyperparameters: Record<string, unknown>) =>
  ({
    id: 'tmpl',
    name: 'saved before the loop fields existed',
    model_id: null,
    dataset_ids: [],
    encoder_type: encoder,
    is_favorite: false,
    hyperparameters: { hidden_dim: 768, latent_dim: 8192, batch_size: 4096, total_steps: 30000, ...hyperparameters },
  }) as unknown as TrainingTemplate;

const load = (tpl: TrainingTemplate) => {
  for (const update of configUpdatesFromTemplate(tpl)) useTrainingsStore.getState().updateConfig(update);
  return useTrainingsStore.getState().config;
};

describe('loading a template into the training store', () => {
  beforeEach(() => {
    useTrainingsStore.getState().resetConfig();
    const { updateConfig } = useTrainingsStore.getState();
    updateConfig({ architecture_type: SAEArchitectureType.STANDARD_SAELENS });
    updateConfig({ learning_rate: 4e-4, lr_decay_steps: 5000, resample_interval: 2500, dead_neuron_threshold: 7000 });
  });

  it("keeps the template's values that a change of architecture resets", () => {
    const config = load(
      template(SAEArchitectureType.JUMPRELU, {
        architecture_type: SAEArchitectureType.JUMPRELU, learning_rate: 1e-5, resample_dead_neurons: false,
      })
    );
    expect(config.architecture_type).toBe(SAEArchitectureType.JUMPRELU);
    expect(config.resample_dead_neurons).toBe(false);
    expect(config.learning_rate).toBe(1e-5);
  });

  it("writes out the loop fields an older template lacks, instead of keeping the form's", () => {
    const config = load(
      template(SAEArchitectureType.STANDARD_SAELENS, { architecture_type: SAEArchitectureType.STANDARD_SAELENS })
    );
    expect(config.lr_decay_steps).toBe(0);
    expect(config.resample_interval).toBe(DEFAULT_RESAMPLE_INTERVAL);
    expect(config.dead_neuron_threshold).toBe(DEFAULT_DEAD_NEURON_THRESHOLD);
  });

  it('gives a TopK template no resampling fields', () => {
    const [, values] = configUpdatesFromTemplate(template(SAEArchitectureType.TOPK, {}));
    for (const field of ['resample_dead_neurons', 'resample_interval', 'dead_neuron_threshold']) {
      expect(field in values, field).toBe(false);
    }
    expect(values.lr_decay_steps).toBe(0);
  });
});

describe('the training form default', () => {
  it('judges a latent dead after the backend schema default of 1,000 steps', () => {
    useTrainingsStore.getState().resetConfig();
    expect(DEFAULT_DEAD_NEURON_THRESHOLD).toBe(1000);
    expect(useTrainingsStore.getState().config.dead_neuron_threshold).toBe(DEFAULT_DEAD_NEURON_THRESHOLD);
  });
});
