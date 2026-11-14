/**
 * Serialising the mixture block of a FEATURE EXTRACTION request.
 *
 * WHY THIS CALLS `serialiseDatasetWeights` INSTEAD OF COPYING IT.
 * The ordering rule is identical to training's and it is the dangerous part:
 * `dataset_weights` is POSITIONAL over the id list, and a weight array built in
 * a different order than the ids is not an error — it silently extracts the
 * wrong mixture, and the server's log line reports the realised split
 * truthfully, which reads like confirmation. One tested implementation, not
 * two that can drift.
 *
 * WHAT DIFFERS FROM TRAINING, AND THE UI COPY MUST SAY SO: the UNIT.
 * Training weights TOKENS, because the optimiser consumes tokens. Extraction
 * weights ROWS, because a top-k example slot is a row. The two also weight
 * different objects — training weights activation extractions, extraction
 * weights tokenizations. Reusing training's 35/30/15/10/10 here is reasonable;
 * describing it as "same as training" is not.
 */

import { serialiseDatasetWeights } from './trainingMixture';

export interface ExtractionMixtureInput {
  /** Selected dataset ids, in the order they will be sent. */
  datasetIds: string[];
  /** Weight per dataset id. Missing entries default to 1. */
  weightsByDataset: Record<string, number> | undefined;
}

export interface ExtractionMixtureBlock {
  dataset_ids?: string[];
  dataset_weights?: number[];
}

/**
 * The mixture fields of an extraction request.
 *
 * `dataset_weights` is omitted — not sent as all-ones — when the operator
 * expressed no preference, because the server's no-weights default is an EVEN
 * split across sources and a uniform array means the same thing. Sending
 * nothing keeps a request from an untouched control byte-identical to the
 * historical single-corpus one.
 */
export function buildExtractionMixture(
  input: ExtractionMixtureInput
): ExtractionMixtureBlock {
  const ids = input.datasetIds ?? [];
  if (ids.length === 0) return {};

  const block: ExtractionMixtureBlock = { dataset_ids: [...ids] };

  const weights = serialiseDatasetWeights(ids, input.weightsByDataset);
  if (weights) block.dataset_weights = weights;

  return block;
}

/**
 * The value for the legacy `dataset_id` query parameter.
 *
 * Always the FIRST selected id, matching what the endpoint stores as
 * `config["dataset_id"]`, so every legacy reader — the extraction list, the
 * dataset-name lookup — keeps showing something coherent rather than null.
 */
export function primaryDatasetId(datasetIds: string[]): string | undefined {
  return datasetIds.length > 0 ? datasetIds[0] : undefined;
}
