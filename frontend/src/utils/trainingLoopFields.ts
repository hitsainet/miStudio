/**
 * The training-loop fields of a training request: learning-rate decay and
 * dead-latent resampling.
 *
 * WHY THIS IS A MODULE. The start-training and save-template payloads were
 * written out by hand, per sparsity type, and they drifted. `resample_dead_neurons`
 * was sent only for L1 frameworks: a JumpReLU request omitted it, the backend
 * default (`True`) applied, and every JumpReLU run from the UI resampled with a
 * routine that was wrong for JumpReLU while the form showed no resampling
 * control at all. `resample_interval` was never sent by either payload, and the
 * template payload dropped `dead_neuron_threshold` too.
 *
 * One function builds the block for both payloads, and it decides which
 * frameworks carry the resampling fields from the same predicate the form uses
 * to show them — so a visible control and a transmitted field cannot disagree.
 */

import { getFrameworkConfig } from '../config/frameworkConfigs';

/** The fields a framework that resamples shows and sends. */
export const RESAMPLING_FIELDS = [
  'resample_dead_neurons',
  'resample_interval',
  'dead_neuron_threshold',
] as const;

/** Backend schema defaults (`TrainingHyperparameters`), for every surface that needs one. */
export const DEFAULT_RESAMPLE_INTERVAL = 5000;
export const DEFAULT_DEAD_NEURON_THRESHOLD = 1000;

/**
 * Whether a framework resamples dead latents.
 *
 * Every framework except TopK, which revives dead latents with its auxiliary
 * loss (Gao et al. 2024) and which the training loop never resamples.
 */
export function usesResampling(architectureType: string): boolean {
  return getFrameworkConfig(architectureType).sparsityType !== 'topk';
}

export interface LoopFieldsInput {
  architecture_type: string;
  lr_decay_steps?: number;
  resample_dead_neurons?: boolean;
  resample_interval?: number;
  dead_neuron_threshold?: number;
}

export interface LoopBlock {
  lr_decay_steps: number;
  resample_dead_neurons?: boolean;
  resample_interval?: number;
  dead_neuron_threshold?: number;
}

const positiveInteger = (value: unknown, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 1 ? Math.floor(value) : fallback;

/** `lr_decay_steps` as the backend accepts it: a non-negative integer, 0 when unset. */
export function normaliseDecaySteps(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.floor(value) : 0;
}

/**
 * The loop fields of a training or template request.
 *
 * `lr_decay_steps` is always sent, 0 included: a template saved without it and
 * later loaded over a form holding a decay would otherwise keep that decay.
 * The resampling fields are sent — all three, the flag included — for every
 * framework that resamples, and never for one that does not.
 */
export function buildLoopBlock(input: LoopFieldsInput): LoopBlock {
  const block: LoopBlock = { lr_decay_steps: normaliseDecaySteps(input.lr_decay_steps) };
  if (!usesResampling(input.architecture_type)) return block;

  const frameworkDefault = getFrameworkConfig(input.architecture_type).defaults.resample_dead_neurons;
  block.resample_dead_neurons =
    typeof input.resample_dead_neurons === 'boolean'
      ? input.resample_dead_neurons
      : frameworkDefault ?? true;
  block.resample_interval = positiveInteger(input.resample_interval, DEFAULT_RESAMPLE_INTERVAL);
  block.dead_neuron_threshold = positiveInteger(
    input.dead_neuron_threshold,
    DEFAULT_DEAD_NEURON_THRESHOLD
  );
  return block;
}

/**
 * The backend's schedule rule, checked before a request is sent:
 * `warmup_steps + lr_decay_steps <= total_steps`. Null when the schedule fits.
 */
export function scheduleError(
  totalSteps: number | undefined,
  warmupSteps: number | undefined,
  decaySteps: number | undefined
): string | null {
  const total = typeof totalSteps === 'number' && Number.isFinite(totalSteps) ? totalSteps : 0;
  const warmup =
    typeof warmupSteps === 'number' && Number.isFinite(warmupSteps) && warmupSteps > 0 ? warmupSteps : 0;
  const decay = normaliseDecaySteps(decaySteps);
  if (warmup + decay > total) {
    return `Warmup steps (${warmup}) plus LR decay steps (${decay}) exceed total steps (${total})`;
  }
  return null;
}
