/**
 * A correlated feature must be named the way the user knows it.
 *
 * The table showed `feat_sae_20260223_131023_01619` — an internal key that
 * appears nowhere else in the product. The modal title says "Feature #13" and
 * the browser lists numbers, so a row keyed by the raw id was unmatchable to
 * anything on screen. The index was available in the service the whole time
 * and was being discarded.
 *
 * MUTATION CONTROLS:
 *   * render feature_id instead of the index -> "names it by number" fails
 *   * invent an index when it is missing     -> "falls back" fails
 */

import { describe, expect, it } from 'vitest';
import { featureLabelFor } from './FeatureCorrelations';

describe('featureLabelFor', () => {
  it('names a feature by its number, matching the rest of the product', () => {
    expect(
      featureLabelFor({
        feature_id: 'feat_sae_20260223_131023_01619',
        neuron_index: 1619,
      })
    ).toBe('#1619');
  });

  it('names index 0 as #0, not as the raw id', () => {
    // `neuron_index: 0` is falsy; a truthiness check here would send the very
    // first feature of every SAE back to showing an internal key.
    expect(featureLabelFor({ feature_id: 'feat_x_0', neuron_index: 0 })).toBe('#0');
  });

  it('falls back to the id when there is no index, rather than inventing one', () => {
    // An older response genuinely does not carry the index. Parsing one out of
    // the id would be a guess that reads exactly like a fact.
    expect(featureLabelFor({ feature_id: 'feat_sae_x_01619' })).toBe(
      'feat_sae_x_01619'
    );
    expect(
      featureLabelFor({ feature_id: 'feat_sae_x_01619', neuron_index: null })
    ).toBe('feat_sae_x_01619');
  });
});
