/**
 * What a probe detects, derived from the dataset it was trained on.
 *
 * ⚠ THE CONFIG NAME IS NOT THE CONCEPT, and using it captioned every probe on the estate
 * with the word "training". The tile read `detects training in LFM2.5-1.2B-Instruct` for
 * nine different probes, none of which detects training — the training view's HuggingFace
 * config happens to be called `training`, and `config || name` picked it up. For the
 * EVALUATION views the same expression is genuinely useful (`anthropic_hh_balanced`
 * distinguishes five views of one repo), which is why the defect hid: the expression is
 * right everywhere it is read except the one place the caption reads it.
 *
 * The concept is the label mapping's POSITIVE side. That is not a heuristic — it is the
 * definition. A probe is fitted to separate the rows mapped to `positive` from the rows
 * mapped to `negative`, so whatever the operator mapped to positive IS what the probe
 * detects. For the models-under-pressure corpus that mapping is
 * `{ "low-stakes": "negative", "high-stakes": "positive" }`, and the caption becomes
 * `detects high-stakes`.
 *
 * Extracted as a pure function on purpose. This repo's recurring failure is a decision
 * buried in JSX where the only available test is a source scrape, and a scrape matches
 * comments and destructuring as happily as calls.
 */
import type { ProbeDataset } from '../../types/probeMonitor';

function labelsMappedTo(mapping: Record<string, string>, target: string): string[] {
  return Object.keys(mapping)
    .filter((label) => String(mapping[label]).toLowerCase() === target)
    .sort();
}

/** The positive labels, sorted so the caption does not reorder between renders. */
export function positiveLabels(dataset: Pick<ProbeDataset, 'label_mapping'>): string[] {
  return labelsMappedTo(dataset.label_mapping ?? {}, 'positive');
}

/**
 * The caption for a probe trained on `dataset`.
 *
 * Falls back to the view's config and then its name when no label maps to positive —
 * which is a real state (a mapping can be empty before a dataset is built), and a wrong
 * caption is better than a blank tile only when it is the view's own name rather than a
 * word invented here.
 */
export function conceptOf(
  dataset: Pick<ProbeDataset, 'label_mapping' | 'config' | 'name'> | null | undefined
): string | null {
  if (!dataset) return null;
  const positives = positiveLabels(dataset);
  if (positives.length === 1) return positives[0];
  // Two labels can both mean the concept ("high-stakes", "critical"). Naming both is
  // honest; silently showing the first would hide half of what the probe was fitted to.
  if (positives.length > 1) return positives.join(' or ');
  return dataset.config || dataset.name || null;
}

/** The three sides of a label mapping, each sorted so a caption does not reorder. */
export interface TrainingLabels {
  /** Raw label values the operator mapped to `positive` — what the probe fires on. */
  positive: string[];
  /** Raw label values mapped to `negative` — what it was fitted to separate them FROM. */
  negative: string[];
  /**
   * Raw label values mapped to `excluded`. These rows were dropped before fitting, so they
   * are not part of what the probe learned — but they are an operator decision, and a
   * caption that omitted them entirely would hide a third of some mappings.
   */
  excluded: string[];
}

/**
 * The raw label values a probe's training was associated with, by side.
 *
 * ⚠ THE VALUES ARE THE CORPUS'S OWN, NOT A VOCABULARY INVENTED HERE. `label_mapping` is
 * `{raw_value: "positive" | "negative" | "excluded"}` exactly as the operator entered it
 * when building the view, so `high-stakes` on the tile is the string that appears in the
 * dataset's label column. Nothing is translated, title-cased or prettified: a reader
 * checking the tile against the corpus must find the same token.
 */
export function trainingLabels(
  dataset: Pick<ProbeDataset, 'label_mapping'> | null | undefined
): TrainingLabels {
  const mapping = dataset?.label_mapping ?? {};
  return {
    positive: labelsMappedTo(mapping, 'positive'),
    negative: labelsMappedTo(mapping, 'negative'),
    excluded: labelsMappedTo(mapping, 'excluded'),
  };
}

/**
 * `"high-stakes vs low-stakes"` — the separation the probe was fitted to make.
 *
 * `null` when either side is empty, because a half-stated separation is misleading in a way
 * a blank is not: "trained on high-stakes" reads as a corpus, not as a contrast, and the
 * contrast is the whole of what a linear probe is. `conceptOf` already carries the positive
 * side alone for the headline, with its own documented fallbacks.
 */
export function labelSeparation(
  dataset: Pick<ProbeDataset, 'label_mapping'> | null | undefined
): string | null {
  const { positive, negative } = trainingLabels(dataset);
  if (positive.length === 0 || negative.length === 0) return null;
  return `${positive.join(' or ')} vs ${negative.join(' or ')}`;
}
