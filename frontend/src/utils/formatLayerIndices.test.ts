/**
 * An extraction's layers must be identifiable from its label.
 *
 * Reported 2026-08-27: "I am not being given an option to choose the correct
 * extractions." The training panel's dropdown rendered "{count}L, {samples}
 * samples", so the OpenWebText-2M extraction at layer 45 and the one at layers
 * 44 and 46 appeared as "1L" and "2L" with no way to tell which was which.
 * Choosing wrong failed only after the request reached the server, leaving a
 * failed training row behind each time.
 */

import { describe, it, expect } from 'vitest';
import { formatLayerIndices } from './formatters';

describe('formatLayerIndices', () => {
  it('names non-adjacent layers individually — the case that caused this', () => {
    expect(formatLayerIndices([44, 46])).toBe('L44, L46');
  });

  it('distinguishes a single layer from a pair', () => {
    expect(formatLayerIndices([45])).toBe('L45');
    expect(formatLayerIndices([44, 46])).not.toBe(formatLayerIndices([45]));
  });

  it('collapses a long consecutive run into a range', () => {
    expect(formatLayerIndices([33, 34, 35, 36, 37])).toBe('L33–L37');
    expect(formatLayerIndices([...Array(16).keys()])).toBe('L0–L15');
  });

  it('keeps a pair expanded rather than calling it a range', () => {
    expect(formatLayerIndices([7, 8])).toBe('L7, L8');
  });

  it('mixes runs and singles', () => {
    expect(formatLayerIndices([1, 2, 3, 9, 20, 21, 22])).toBe(
      'L1–L3, L9, L20–L22',
    );
  });

  it('sorts and de-duplicates', () => {
    expect(formatLayerIndices([46, 44, 44])).toBe('L44, L46');
  });

  it('says so when there are none, rather than rendering nothing', () => {
    expect(formatLayerIndices([])).toBe('no layers');
    expect(formatLayerIndices(undefined)).toBe('no layers');
    expect(formatLayerIndices(null)).toBe('no layers');
  });
});
