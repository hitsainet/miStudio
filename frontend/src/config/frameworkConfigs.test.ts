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
import type { TrainingConfig } from '../stores/trainingsStore';
import { RESAMPLING_FIELDS, usesResampling } from '../utils/trainingLoopFields';
import { buildTrainingHyperparameters } from '../utils/trainingHyperparameters';

// Vite's `?raw` import rather than node's `fs`: the test tsconfig has no node
// types, and adding them to clear five errors would raise the down-only
// type-check ratchet for everyone.
import panelSource from '../components/panels/TrainingPanel.tsx?raw';

/**
 * WHY THIS CALLS THE BUILDER INSTEAD OF READING THE PANEL'S SOURCE (review round 3,
 * R3-C). This guard used to split `TrainingPanel.tsx` on `sparsityType === 'l1' && {`
 * and read the keys of each block. Round 3 moved both payloads onto one builder,
 * `utils/trainingHyperparameters.ts`, and the scrape went red for the harmless reason
 * that the text moved. Pointing the scrape at the new file would have passed for the
 * wrong reason: a block's keys prove a name is written, not that the request carries
 * the value.
 *
 * So the guard is now two links, each asserted where it lives:
 *   1. here, the builder sends every visible sparsity field with the value the form
 *      holds, for every framework;
 *   2. in `components/panels/TrainingPanel.r3c.test.tsx`, Start Training and Save as
 *      Template both send the builder's output (controls M1 and M2 there revert
 *      either site to its old hand-written list and turn it red).
 *
 * MUTATION CONTROLS (results in the R3-C record): P5 drops `target_l0` from the
 * JumpReLU block, P6 drops `ste_bandwidth`, P7 drops `aux_k` from the TopK block.
 */
const SPARSITY_SPECIFIC = [
  'l1_alpha', 'target_l0', 'sparsity_coeff', 'initial_threshold',
  'bandwidth', 'ste_bandwidth', 'top_k', 'aux_k', 'aux_loss_alpha',
] as const;

/** A distinct value per field, so a field sent with ANOTHER field's value is caught too. */
const SENTINELS: Record<string, number> = Object.fromEntries(
  SPARSITY_SPECIFIC.map((field, index) => [field, 0.101 + index / 1000])
);

describe('every visible field is actually transmitted', () => {
  for (const [arch, cfg] of Object.entries(FRAMEWORK_CONFIGS)) {
    it(`${arch}: no visible sparsity field is dropped from the payload`, () => {
      const expected = cfg.visibleFields.filter((f) =>
        (SPARSITY_SPECIFIC as readonly string[]).includes(f)
      );
      // Every framework shows at least one; an empty list would pass vacuously.
      expect(expected.length, `${arch} shows no sparsity field`).toBeGreaterThan(0);

      const config = {
        model_id: 'm', dataset_ids: ['ds'], hidden_dim: 64, latent_dim: 512,
        architecture_type: arch as SAEArchitectureType, training_layers: [0], hook_types: ['residual'],
        learning_rate: 1e-4, batch_size: 32, total_steps: 100,
        ...SENTINELS,
      } as TrainingConfig;
      const sent = buildTrainingHyperparameters(config, {
        sourceIds: ['ds'],
        weightsBySource: undefined,
      }) as unknown as Record<string, unknown>;

      const dropped = expected.filter((field) => sent[field] !== SENTINELS[field]);
      expect(dropped, `${arch} drops: ${dropped.join(', ')}`).toEqual([]);
    });
  }
});

describe('resampling controls follow the frameworks that resample', () => {
  // The payload sends the resampling fields for exactly `usesResampling`
  // (see trainingLoopFields.test.ts). A framework that shows them must send
  // them, and one that sends them must show them.
  for (const arch of Object.keys(FRAMEWORK_CONFIGS)) {
    it(`${arch}: shown if and only if it resamples`, () => {
      for (const field of RESAMPLING_FIELDS) {
        expect(isFieldVisible(arch, field), field).toBe(usesResampling(arch));
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

  it('exposes the resampling controls, which a JumpReLU run acts on', () => {
    // They were hidden on the claim that JumpReLU does not resample. It did —
    // the payload omitted the flag and the backend default is true — so the
    // controls that governed every run were the ones a user could not see.
    for (const field of RESAMPLING_FIELDS) {
      expect(isFieldVisible(SAEArchitectureType.JUMPRELU, field), field).toBe(true);
      expect(jump.hiddenFields).not.toContain(field);
    }
    expect(jump.defaults.resample_dead_neurons).toBe(true);
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
