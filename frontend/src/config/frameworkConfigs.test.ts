/**
 * A visible control that is never transmitted is worse than a hidden one.
 *
 * WHY THIS FILE EXISTS. Round 4 found `target_l0` had been moved from
 * `hiddenFields` to `visibleFields` for JumpReLU — with a comment calling it
 * "the highest-leverage sparsity knob JumpReLU has" — while the request payload
 * for `sparsityType === 'l0'` never included it. The control rendered, accepted
 * edits, and sent nothing. An honest absence was converted into a silent lie,
 * and the backend went on reading its 0.05 default.
 *
 * `ste_bandwidth` was worse: added to defaults and visibleFields with no input
 * rendered anywhere and no payload entry.
 *
 * No frontend test referenced `frameworkConfigs` or `isFieldVisible` at all.
 */

import { describe, it, expect } from 'vitest';
import { FRAMEWORK_CONFIGS, isFieldVisible } from './frameworkConfigs';
import { SAEArchitectureType } from '../types/training';

// Vite's `?raw` import rather than node's `fs`: the test tsconfig has no node
// types, and adding them to clear five errors would raise the down-only
// type-check ratchet for everyone.
import panelSource from '../components/panels/TrainingPanel.tsx?raw';

/**
 * The payload spreads for one sparsity type — one array PER BLOCK.
 *
 * Deliberately not unioned. There are two spreads (start-training and
 * save-template) and the first version of this helper merged them, so a field
 * present in only one satisfied the assertion. That is the same
 * "satisfied by the wrong occurrence" failure that let this defect through in
 * the first place; every block must carry the field.
 */
function payloadBlocksFor(sparsityType: string): string[][] {
  const blocks = panelSource.split(`sparsityType === '${sparsityType}' && {`);
  expect(blocks.length).toBeGreaterThan(1);
  return blocks.slice(1).map((b: string) => {
    const body = b.slice(0, b.indexOf('}'));
    return Array.from(body.matchAll(/^\s*(\w+):/gm)).map((m) => String(m[1]));
  });
}

describe('every visible field is actually transmitted', () => {
  const byType: Record<string, string> = {
    l1: 'l1',
    l0: 'l0',
    topk: 'topk',
  };

  for (const [arch, cfg] of Object.entries(FRAMEWORK_CONFIGS)) {
    const sparsityType = byType[cfg.sparsityType];
    if (!sparsityType) continue;

    it(`${arch}: no visible sparsity field is dropped from ANY payload`, () => {
      // Fields owned by the sparsity-specific spread. Everything else is in
      // the common block and is not this test's concern.
      const sparsitySpecific = [
        'l1_alpha', 'target_l0', 'sparsity_coeff', 'initial_threshold',
        'bandwidth', 'ste_bandwidth', 'top_k', 'aux_k', 'aux_loss_alpha',
      ];
      const expected = cfg.visibleFields.filter((f) => sparsitySpecific.includes(f));

      for (const [i, block] of payloadBlocksFor(sparsityType).entries()) {
        const sent = new Set(block);
        const dropped = expected.filter((f) => !sent.has(f));
        expect(dropped, `payload block ${i} drops: ${dropped.join(', ')}`).toEqual([]);
      }
    });
  }
});

describe('JumpReLU specifically', () => {
  const jump = FRAMEWORK_CONFIGS[SAEArchitectureType.JUMPRELU];

  it('exposes target_l0, which sets every threshold at step 0', () => {
    expect(jump.visibleFields).toContain('target_l0');
    expect(isFieldVisible(SAEArchitectureType.JUMPRELU, 'target_l0')).toBe(true);
  });

  it('exposes ste_bandwidth, the only dead-latent revival path it has', () => {
    expect(jump.visibleFields).toContain('ste_bandwidth');
  });

  it('hides dead_neuron_threshold, which does nothing for this framework', () => {
    // It is read only inside the resampling block, which JumpReLU disables;
    // detection uses an EMA instead. Showing it beside a knob that could not
    // be set was exactly backwards.
    expect(jump.hiddenFields).toContain('dead_neuron_threshold');
    expect(isFieldVisible(SAEArchitectureType.JUMPRELU, 'dead_neuron_threshold')).toBe(false);
  });

  it('renders a control BOUND TO STATE for every visible sparsity field', () => {
    // Asserting on `updateConfig({ field:` rather than an element id: the id
    // convention is inconsistent (`initial-threshold` vs `target_l0`), and an
    // id proves an element exists while this proves it writes to the config
    // that the payload is built from. `ste_bandwidth` shipped with defaults,
    // visibleFields and NO input at all.
    for (const field of ['target_l0', 'ste_bandwidth', 'bandwidth', 'initial_threshold']) {
      expect(panelSource).toContain(`updateConfig({ ${field}:`);
    }
  });

  it('carries defaults for the knobs it exposes', () => {
    expect(jump.defaults.target_l0).toBeDefined();
    expect(jump.defaults.ste_bandwidth).toBeDefined();
  });
});
