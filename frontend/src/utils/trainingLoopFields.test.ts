/**
 * The loop fields every training and template request carries.
 *
 * The regression this pins: a JumpReLU request omitted `resample_dead_neurons`,
 * so the backend default (true) silently applied while the form showed no
 * resampling control, and no request ever carried `resample_interval`.
 *
 * MUTATION CONTROLS (2026-09-15, WS-LOOP; each applied alone by a runner that
 * checked the target occurred once, ran the named vitest files, restored the
 * bytes and verified sha256). All seven went red:
 *   F1 start-training payload drops `...buildLoopBlock(config)`
 *        -> TrainingPanel.test: "sends the resampling fields and LR decay for a
 *           JumpReLU run", "should call createTraining with correct config"
 *   F2 save-template payload drops `...buildLoopBlock(config)`
 *        -> TrainingPanel.test: "saves a JumpReLU template with the resampling fields"
 *   F3 `usesResampling` true for TopK too
 *        -> this file (2), frameworkConfigs.test "topk: shown if and only if it
 *           resamples", TrainingTemplateForm.test "neither shows nor sends ... TopK"
 *   F4 `buildLoopBlock` stops setting `resample_dead_neurons`
 *        -> this file (3), TrainingPanel.test (3), TrainingTemplateForm.test (2)
 *   F5 the template form's payload drops its `buildLoopBlock` spread
 *        -> TrainingTemplateForm.test (3)
 *   F6 the panel no longer checks the schedule before sending
 *        -> TrainingPanel.test "refuses a schedule whose warmup and decay exceed the run"
 *   F7 JumpReLU's visibleFields lose the three resampling controls
 *        -> frameworkConfigs.test "exposes the resampling controls", "jumprelu:
 *           shown if and only if it resamples"
 */

import { describe, it, expect } from 'vitest';
import {
  RESAMPLING_FIELDS,
  buildLoopBlock,
  normaliseDecaySteps,
  scheduleError,
  usesResampling,
} from './trainingLoopFields';
import { FRAMEWORK_CONFIGS } from '../config/frameworkConfigs';
import { SAEArchitectureType } from '../types/training';

describe('usesResampling', () => {
  it('is every framework except TopK', () => {
    for (const arch of Object.keys(FRAMEWORK_CONFIGS)) {
      expect(usesResampling(arch), arch).toBe(arch !== SAEArchitectureType.TOPK);
    }
  });
});

describe('buildLoopBlock', () => {
  it('sends the flag, interval and threshold for JumpReLU', () => {
    expect(
      buildLoopBlock({
        architecture_type: SAEArchitectureType.JUMPRELU,
        resample_dead_neurons: false,
        resample_interval: 2500,
        dead_neuron_threshold: 7000,
        lr_decay_steps: 4000,
      })
    ).toEqual({
      lr_decay_steps: 4000,
      resample_dead_neurons: false,
      resample_interval: 2500,
      dead_neuron_threshold: 7000,
    });
  });

  it('sends an explicit flag even when the form never set one', () => {
    const block = buildLoopBlock({ architecture_type: SAEArchitectureType.JUMPRELU });
    expect(block.resample_dead_neurons).toBe(
      FRAMEWORK_CONFIGS[SAEArchitectureType.JUMPRELU].defaults.resample_dead_neurons
    );
    expect(block.resample_interval).toBe(5000);
    expect(block.dead_neuron_threshold).toBe(1000);
  });

  it('sends every resampling field for every framework that resamples, and none for TopK', () => {
    for (const arch of Object.keys(FRAMEWORK_CONFIGS)) {
      const block = buildLoopBlock({ architecture_type: arch, resample_dead_neurons: true });
      for (const field of RESAMPLING_FIELDS) {
        expect(field in block, `${arch}.${field}`).toBe(usesResampling(arch));
      }
    }
  });

  it('always sends lr_decay_steps, as a non-negative integer', () => {
    expect(buildLoopBlock({ architecture_type: SAEArchitectureType.TOPK })).toEqual({ lr_decay_steps: 0 });
    expect(normaliseDecaySteps(Number.NaN)).toBe(0);
    expect(normaliseDecaySteps(-5)).toBe(0);
    expect(normaliseDecaySteps(1500.7)).toBe(1500);
  });

  it('replaces an unusable interval or threshold with the backend default', () => {
    const block = buildLoopBlock({
      architecture_type: SAEArchitectureType.STANDARD_SAELENS,
      resample_interval: Number.NaN,
      dead_neuron_threshold: 0,
    });
    expect(block.resample_interval).toBe(5000);
    expect(block.dead_neuron_threshold).toBe(1000);
  });
});

describe('scheduleError', () => {
  it('accepts warmup plus decay up to total steps exactly', () => {
    expect(scheduleError(10000, 2000, 8000)).toBeNull();
    expect(scheduleError(10000, 2000, 0)).toBeNull();
  });

  it('refuses warmup plus decay beyond total steps', () => {
    expect(scheduleError(10000, 2000, 8001)).toMatch(/exceed total steps/);
  });
});
