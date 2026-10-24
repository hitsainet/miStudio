/**
 * What the training card's "dead" count measures, stated honestly (review round 1, R1-D L2).
 *
 * The count stored as `dead_neurons` is NOT the resampler's definition. The training
 * loop keeps a per-latent activity estimate, `ema = ema * (1 - batch_size / 50,000) + fired`,
 * and counts latents whose estimate is below 0.01 (`workers/training_tasks.py`). A
 * latent that fired on every step reads as inactive after only a few dozen silent
 * steps at a large batch — 84 at batch 4,096 — while resampling waits for
 * `dead_neuron_threshold` consecutive silent steps (1,000 by default). The card
 * labelled the first as "Dead Neurons" beside a control that sets the second.
 *
 * The stored metric keeps its meaning (historical rows are not restated); the card
 * names it for what it is and shows both windows.
 *
 * KEEP IN STEP WITH `ema_window_tokens` and the `< 0.01` test in the training loop.
 */

export const ACTIVITY_EMA_WINDOW_TOKENS = 50_000;
export const ACTIVITY_EMA_INACTIVE_BELOW = 0.01;

/**
 * Silent steps after which a latent that had fired on EVERY step counts as inactive.
 * A latent that fired less often starts lower and crosses sooner, so this is an upper
 * bound. Null when the batch size is unknown.
 */
export function inactivityWindowSteps(batchSize: number | undefined | null): number | null {
  if (typeof batchSize !== 'number' || !Number.isFinite(batchSize) || batchSize <= 0) return null;
  const decay = Math.max(0, 1 - batchSize / ACTIVITY_EMA_WINDOW_TOKENS);
  if (decay === 0) return 1;
  const steadyState = 1 / (1 - decay);
  return Math.ceil(Math.log(ACTIVITY_EMA_INACTIVE_BELOW / steadyState) / Math.log(decay));
}

/** The card's tooltip for the count: both windows, in steps. */
export function inactiveLatentsExplanation(
  batchSize: number | undefined | null,
  deadNeuronThreshold: number | undefined | null
): string {
  const window = inactivityWindowSteps(batchSize);
  const measured =
    window === null
      ? 'Latents with no recent firing, from a fast activity estimate.'
      : `Latents with no firing in about the last ${window.toLocaleString()} steps (a fast activity estimate).`;
  const threshold =
    typeof deadNeuronThreshold === 'number' && deadNeuronThreshold > 0
      ? `Resampling treats a latent as dead only after ${deadNeuronThreshold.toLocaleString()} consecutive silent steps.`
      : 'Resampling uses its own threshold, dead_neuron_threshold.';
  return `${measured} ${threshold}`;
}
