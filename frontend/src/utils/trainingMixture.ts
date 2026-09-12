/**
 * Serialising the mixture block of a training request.
 *
 * WHY THIS IS A MODULE AND NOT THREE LINES IN `handleStartTraining`.
 * `dataset_weights` is POSITIONAL: the worker builds its `extractions` list by
 * iterating the `extraction_ids` it was given, in order, and then indexes
 * `dataset_weights[i]` against `extractions[i]`. Nothing validates the pairing.
 * A weight array built in a different order than the ids is not an error — it
 * silently trains on the wrong mixture, and the log line reports the realised
 * split truthfully, which reads like confirmation.
 *
 * So the UI keeps weights KEYED BY EXTRACTION ID and this function is the only
 * place that flattens them to an array. The ordering lives in one tested
 * function rather than in a component that also renders.
 */

export interface MixtureInput {
  /** The selected extraction ids, in the order they will be sent. */
  extractionIds: string[] | undefined;
  /** Weight per extraction id. Missing entries default to 1. */
  weightsByExtraction: Record<string, number> | undefined;
  holdoutFraction: number | undefined;
}

export interface MixtureBlock {
  dataset_weights?: number[];
  holdout_fraction?: number;
}

/** A weight the backend will accept: finite, non-negative. */
const isUsableWeight = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0;

/**
 * Weights in `extractionIds` order, or undefined when they should be omitted.
 *
 * Omitted — not sent as all-ones — when the operator has expressed no
 * preference, because omitting means "proportional to real tokens" while a
 * uniform array means "equal shares regardless of size". Those are different
 * mixtures, and the second is very unlikely to be what someone who never
 * touched the control wanted.
 */
export function serialiseDatasetWeights(
  extractionIds: string[] | undefined,
  weightsByExtraction: Record<string, number> | undefined
): number[] | undefined {
  if (!extractionIds || extractionIds.length === 0) return undefined;
  if (!weightsByExtraction) return undefined;

  const weights = extractionIds.map((id) => {
    const value = weightsByExtraction[id];
    return isUsableWeight(value) ? value : 1;
  });

  // All equal carries no information the default does not already express.
  const allEqual = weights.every((w) => w === weights[0]);
  if (allEqual) return undefined;

  // Every source at zero would ask for an empty corpus; the backend normalises
  // by the sum and would divide by zero. Treat it as no preference.
  if (weights.every((w) => w === 0)) return undefined;

  return weights;
}

/**
 * The mixture fields of a training request.
 *
 * `holdout_fraction` is omitted at 0 rather than sent, so a request from a
 * caller that never set it is byte-identical to the historical one.
 */
export function buildMixtureBlock(input: MixtureInput): MixtureBlock {
  const block: MixtureBlock = {};

  const weights = serialiseDatasetWeights(
    input.extractionIds,
    input.weightsByExtraction
  );
  if (weights) block.dataset_weights = weights;

  const holdout = input.holdoutFraction;
  if (
    typeof holdout === 'number' &&
    Number.isFinite(holdout) &&
    holdout > 0 &&
    holdout < 1
  ) {
    block.holdout_fraction = holdout;
  }

  return block;
}
