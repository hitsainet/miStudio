/**
 * The training card's history reads aggregate metric rows only (review R1-C).
 *
 * NEGATIVE CONTROL (applied alone, this file run, restored, sha256 verified):
 *   UI12 aggregateMetricRows returns every row   -> RED  keeps only the aggregate rows
 *
 * Review R1-A, A5: a multi-hook run's SAEs share a layer, and both hooks' rows carry it.
 */
import { describe, expect, it } from 'vitest';
import { aggregateMetricRows } from './trainingMetricRows';

describe('aggregateMetricRows', () => {
  const step = (s: number, layer_idx: number | null | undefined, fvu_centred: number) => ({
    step: s, layer_idx, fvu_centred,
  });

  it('keeps only the aggregate rows, in order', () => {
    const rows = [
      step(100, 11, 0.9),      // one SAE
      step(100, -12, 0.8),     // held-out, layer 11
      step(100, -1011, 0.7),   // a legacy spliced-CE row
      step(100, null, 0.31),   // the aggregate
      step(200, null, 0.29),
      step(200, 12, 0.95),
    ];
    expect(aggregateMetricRows(rows)).toEqual([step(100, null, 0.31), step(200, null, 0.29)]);
  });

  it('treats a row served without layer_idx as aggregate, as the endpoint once served every row', () => {
    const rows: { step: number; fvu_centred: number; layer_idx?: number | null }[] = [{ step: 1, fvu_centred: 0.4 }];
    expect(aggregateMetricRows(rows)).toEqual(rows);
  });

  it('keeps only the aggregate row of a multi-hook run, whose SAEs share a layer', () => {
    const hooked = (s: number, layer_idx: number | null, hook_type: string | null, fvu_centred: number) => ({
      step: s, layer_idx, hook_type, fvu_centred,
    });
    const rows = [
      hooked(100, 11, 'residual', 0.9),
      hooked(100, 11, 'mlp', 0.7),
      hooked(100, -12, 'residual', 0.8),
      hooked(100, -12, 'mlp', 0.6),
      hooked(100, null, null, 0.31),
    ];
    expect(aggregateMetricRows(rows)).toEqual([hooked(100, null, null, 0.31)]);
  });
});
