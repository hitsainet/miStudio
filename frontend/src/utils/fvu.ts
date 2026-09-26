/**
 * Which FVU to show, and what to call it (SAE training remediation, item 5).
 *
 * miStudio stored one FVU until 2026-09-15: var(x - x_hat) / var(x) over every
 * element with ONE global mean. On activations with large near-constant
 * dimensions it reads low (0.26 vs 0.32 at LFM2.5-1.2B layer 11). The standard,
 * per-dimension-centred FVU is now stored beside it.
 *
 * The rule, in one place so the card, the chart and the log line cannot disagree:
 *   - the centred value, when present, is shown as "FVU";
 *   - otherwise the legacy value is shown, labelled "FVU (legacy)" — runs
 *     recorded before the change have only that value, and presenting it under
 *     the plain name would let an old run look ~0.05 better than a new one.
 */

export const FVU_TITLE =
  'FVU: sum|x - x_hat|^2 / sum|x - mean|^2, with the mean taken per dimension. ' +
  '0 is a perfect reconstruction; 1 is no better than predicting the mean.';

export const LEGACY_FVU_TITLE =
  'Legacy FVU: var(x - x_hat) / var(x) with one global mean. It reads low on activations ' +
  'with large constant dimensions, so it is not comparable with FVU. Runs recorded before ' +
  '2026-09-15 have only this value.';

export interface FvuDisplay {
  /** The value to show, or null when neither is reported. */
  value: number | null;
  /** "FVU" for the centred value, "FVU (legacy)" when only the legacy value exists. */
  label: string;
  /** Tooltip explaining the value shown. */
  title: string;
  /** The legacy value, when it is shown beside the centred one. */
  legacy: number | null;
  isLegacyOnly: boolean;
}

function finite(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function fvuDisplay(
  centred: number | null | undefined,
  legacy: number | null | undefined,
): FvuDisplay {
  const c = finite(centred);
  const l = finite(legacy);
  if (c !== null) {
    return { value: c, label: 'FVU', title: FVU_TITLE, legacy: l, isLegacyOnly: false };
  }
  if (l !== null) {
    return { value: l, label: 'FVU (legacy)', title: LEGACY_FVU_TITLE, legacy: l, isLegacyOnly: true };
  }
  return { value: null, label: 'FVU', title: FVU_TITLE, legacy: null, isLegacyOnly: false };
}

/**
 * The series to chart: the centred history when any point has one, otherwise the
 * legacy history. Never a mix — the two scales differ, so splicing them would
 * draw a step that is not in the training.
 */
export function fvuSeries(centred: number[], legacy: number[]): { series: number[]; isLegacy: boolean } {
  if (centred.some(Number.isFinite)) {
    return { series: centred, isLegacy: false };
  }
  return { series: legacy, isLegacy: legacy.some(Number.isFinite) };
}
