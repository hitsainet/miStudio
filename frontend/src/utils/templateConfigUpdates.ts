/**
 * The training-form updates that load a template, in the order they must be applied.
 *
 * WHY TWO UPDATES (review round 1, R1-A). `updateConfig` merges an update and
 * then, when the update changes the architecture, overwrites that framework's
 * defaults on top — the learning rate, the sparsity settings and
 * `resample_dead_neurons` among them. Loading a template by spreading its
 * hyperparameters into ONE update therefore lost exactly those values whenever
 * the template's architecture differed from the form's: a JumpReLU template saved
 * with resampling off loaded into a Standard form with it on, and the run
 * resampled.
 *
 * So the architecture goes first, alone (its defaults apply), then the template's
 * values over them.
 *
 * WHY THE LOOP FIELDS ARE WRITTEN OUT. A template saved before `lr_decay_steps`,
 * `resample_interval` or `dead_neuron_threshold` was sent carries none of them,
 * and a spread left the form's own values in place — a decay from an earlier edit
 * rode into the next run unseen. Missing, they take the backend's defaults.
 */

import type { TrainingConfig } from '../stores/trainingsStore';
import type { SAEArchitectureType } from '../types/training';
import type { TrainingTemplate } from '../types/trainingTemplate';
import { getFrameworkConfig } from '../config/frameworkConfigs';
import {
  DEFAULT_DEAD_NEURON_THRESHOLD,
  DEFAULT_RESAMPLE_INTERVAL,
  normaliseDecaySteps,
  usesResampling,
} from './trainingLoopFields';

export function configUpdatesFromTemplate(template: TrainingTemplate): Partial<TrainingConfig>[] {
  const hp = template.hyperparameters;
  const architecture = template.encoder_type as SAEArchitectureType;
  const loop: Partial<TrainingConfig> = { lr_decay_steps: normaliseDecaySteps(hp.lr_decay_steps) };
  if (usesResampling(architecture)) {
    loop.resample_dead_neurons =
      hp.resample_dead_neurons ?? getFrameworkConfig(architecture).defaults.resample_dead_neurons ?? true;
    loop.resample_interval = hp.resample_interval ?? DEFAULT_RESAMPLE_INTERVAL;
    loop.dead_neuron_threshold = hp.dead_neuron_threshold ?? DEFAULT_DEAD_NEURON_THRESHOLD;
  }
  return [
    { architecture_type: architecture },
    { ...(hp as Partial<TrainingConfig>), architecture_type: architecture, ...loop },
  ];
}
