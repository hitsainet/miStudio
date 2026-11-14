/**
 * The overlay that a Templates-panel save sends (R2D-4, review round 2 lane R2-F).
 *
 * The form-level tests in TrainingTemplateForm.r2f.test.tsx drive the real form.
 * These pin the three rules of the decision itself.
 *
 * Mutation controls (all red, restored by bytes, sha256 verified; full table in
 * .claude/context/sessions/review_sae_remediation_R2_F_2026-09-15.md):
 *   F1 the stored keys dropped (`const result = {}` never filled from `stored`)
 *   F2 the stored value wins over the edit (the two loops swapped)
 *   F3 a cleared field resurrected (an `undefined` edit keeps the stored value)
 *   F4 the resampling fields not owned (`RESAMPLING_FIELDS` left out of `owned`)
 *   F7 the dataset tail dropped (`[id]` instead of `[id, ...rest]`)
 */

import { describe, it, expect } from 'vitest';
import { overlayDatasetIds, overlayHyperparameters } from './templateFormPayload';

describe('overlayHyperparameters', () => {
  const stored = {
    training_layers: [11, 12, 13],
    sparsity_warmup_steps: 10000,
    total_steps: 50000,
    grad_clip_norm: 1.0,
    resample_interval: 5000,
    dead_neuron_threshold: 10000,
    resample_dead_neurons: true,
  };

  it('keeps every stored key the form does not write, unchanged', () => {
    const out = overlayHyperparameters(stored, { total_steps: 150000 });
    expect(out.training_layers).toEqual([11, 12, 13]);
    expect(out.sparsity_warmup_steps).toBe(10000);
  });

  it('takes the edited value for a key the form writes', () => {
    expect(overlayHyperparameters(stored, { total_steps: 150000 }).total_steps).toBe(150000);
  });

  it('removes a key the form clears, and never takes it from the stored template', () => {
    const out = overlayHyperparameters(stored, { grad_clip_norm: undefined });
    expect('grad_clip_norm' in out).toBe(false);
  });

  it('owns the resampling fields even when the form leaves them out (TopK)', () => {
    const out = overlayHyperparameters(stored, { architecture_type: 'topk', total_steps: 50000 });
    for (const field of ['resample_dead_neurons', 'resample_interval', 'dead_neuron_threshold']) {
      expect(field in out, field).toBe(false);
    }
    expect(out.training_layers).toEqual([11, 12, 13]);
  });

  it('is the form values alone when there is no stored template', () => {
    expect(overlayHyperparameters(undefined, { total_steps: 1, l1_alpha: undefined })).toStrictEqual({
      total_steps: 1,
    });
  });
});

describe('overlayDatasetIds', () => {
  it('keeps the datasets after the first', () => {
    expect(overlayDatasetIds(['ds_a', 'ds_b', 'ds_c'], 'ds_a')).toEqual(['ds_a', 'ds_b', 'ds_c']);
    expect(overlayDatasetIds(['ds_a', 'ds_b', 'ds_c'], ' ds_z ')).toEqual(['ds_z', 'ds_b', 'ds_c']);
  });

  it('removes only the first dataset when the field is cleared', () => {
    expect(overlayDatasetIds(['ds_a', 'ds_b'], '')).toEqual(['ds_b']);
    expect(overlayDatasetIds(undefined, '')).toEqual([]);
    expect(overlayDatasetIds([], 'ds_legacy')).toEqual(['ds_legacy']);
  });
});
