/**
 * What a Templates-panel save sends: the template's stored values, with the form's
 * edits applied over them.
 *
 * WHY (review round 2, R2D-4 / R2-F, 2026-09-15). The form built `hyperparameters`
 * from its own twenty fields, and the backend's update replaces the whole dict. So
 * opening a template to change one number and pressing Update Template deleted every
 * key the form has no control for. Template 6a460fbe, the planned 16K JumpReLU run,
 * lost `training_layers` [11, 12, 13], `hook_types`, `sparsity_warmup_steps` 10,000,
 * `normalize_activations`, `normalize_decoder` and `evaluate_ce_delta`. The run would
 * then have trained on whatever the training panel happened to hold.
 *
 * WHY THE PRESERVATION IS HERE AND NOT A BACKEND MERGE. Only the form knows which
 * keys it shows, so only the form can tell "the user cleared this" from "the form
 * never had this". A backend merge would bring back a value the user cleared. It
 * would also validate the request, not the dict it stores, and the schedule rule
 * (`warmup + decay <= total`) spans fields. The backend update therefore keeps its
 * replace semantics, and the client sends the whole dict.
 */

import { RESAMPLING_FIELDS } from '../../utils/trainingLoopFields';

export type HyperparameterRecord = Record<string, unknown>;

/**
 * The form's hyperparameters applied over the template's stored ones.
 *
 * - A key the form does not own keeps its stored value, unchanged.
 * - A key the form owns takes the form's value.
 * - A key the form owns but leaves `undefined` (a cleared optional field) is
 *   removed, never taken from the stored template. The backend then stores its
 *   schema default, as a create does.
 *
 * The form owns every key it writes, plus the resampling fields. `buildLoopBlock`
 * leaves those out for TopK, so a TopK save must not carry a stored value the form
 * neither shows nor sends.
 */
export function overlayHyperparameters(
  stored: object | null | undefined,
  formValues: HyperparameterRecord
): HyperparameterRecord {
  const owned = new Set<string>([...Object.keys(formValues), ...RESAMPLING_FIELDS]);
  const result: HyperparameterRecord = {};
  for (const [key, value] of Object.entries(stored ?? {})) {
    if (!owned.has(key)) result[key] = value;
  }
  for (const [key, value] of Object.entries(formValues)) {
    if (value !== undefined) result[key] = value;
  }
  return result;
}

/**
 * `dataset_ids` for a save. The form shows one dataset: the template's first.
 *
 * The rest are kept. An edited id replaces the first entry. A cleared field
 * removes the first entry and keeps the others.
 */
export function overlayDatasetIds(
  storedIds: readonly string[] | null | undefined,
  shownId: string
): string[] {
  const rest = (storedIds ?? []).slice(1);
  const id = shownId.trim();
  return id ? [id, ...rest] : rest;
}
