/**
 * Which FVU is shown, under which name (SAE training remediation, item 5).
 *
 * The legacy value reads ~0.05 low on real residual streams. Shown under the
 * plain name beside a new run's centred value, an old run would look better than
 * it is — so the label is part of the contract, not decoration.
 *
 * MUTATION CONTROLS (2026-09-15; each applied alone, source restored and verified by
 * sha256). All went red:
 *   UI1 legacy-only label 'FVU (legacy)' -> 'FVU'          -> labels a run...legacy, never presents the legacy value...,
 *                                                            TrainingCard.test "labels a run recorded before the change as legacy"
 *   UI2 centred preferred only when legacy is absent      -> shows the centred value as "FVU"..., prefers the centred value...,
 *                                                            TrainingCard.test "shows the centred FVU as FVU..."
 *   UI3 card calls fvuDisplay(current_fvu, null)          -> both TrainingCard.test FVU label tests
 *   UI4 evaluation panel not rendered on completed cards  -> TrainingCard.test "renders on a completed training...", "shows a recorded result"
 *   UI5 Evaluate not wired to the store                   -> TrainingCard.test "renders on a completed training and sends Evaluate..."
 *   UI6 WS maps data.fvu into current_fvu_centred         -> useTrainingWebSocket.test "should carry the centred FVU beside the legacy one..."
 *   UI9 training:evaluation handler not registered        -> useTrainingWebSocket.test registration (2) and training:evaluation (2) tests
 */

import { describe, expect, it } from 'vitest';
import { fvuDisplay, fvuSeries } from './fvu';

describe('fvuDisplay', () => {
  it('shows the centred value as "FVU", with the legacy value alongside', () => {
    const shown = fvuDisplay(0.32, 0.26);
    expect(shown).toMatchObject({ value: 0.32, label: 'FVU', legacy: 0.26, isLegacyOnly: false });
  });

  it('labels a run that has only the legacy value as legacy', () => {
    const shown = fvuDisplay(null, 0.26);
    expect(shown).toMatchObject({ value: 0.26, label: 'FVU (legacy)', isLegacyOnly: true });
    expect(shown.title).toMatch(/legacy/i);
  });

  it('never presents the legacy value under the plain name', () => {
    expect(fvuDisplay(undefined, 0.1).label).not.toBe('FVU');
    expect(fvuDisplay(Number.NaN, 0.1).label).not.toBe('FVU');
  });

  it('reports nothing, not zero, when neither value exists', () => {
    expect(fvuDisplay(null, null)).toMatchObject({ value: null, legacy: null, isLegacyOnly: false });
  });

  it('prefers the centred value even when it is larger', () => {
    expect(fvuDisplay(0.9, 0.1).value).toBe(0.9);
  });
});

describe('fvuSeries', () => {
  it('charts the centred history when any point has one', () => {
    const result = fvuSeries([Number.NaN, 0.3, 0.29], [0.25, 0.24, 0.23]);
    expect(result.isLegacy).toBe(false);
    expect(result.series).toEqual([Number.NaN, 0.3, 0.29]);
  });

  it('falls back to the legacy history, marked, and never mixes the two', () => {
    const result = fvuSeries([Number.NaN, Number.NaN], [0.25, 0.24]);
    expect(result).toEqual({ series: [0.25, 0.24], isLegacy: true });
  });

  it('marks nothing as legacy when neither series has a value', () => {
    expect(fvuSeries([Number.NaN], [Number.NaN]).isLegacy).toBe(false);
  });
});
