import { describe, it, expect } from 'vitest';
import {
  composeSAEOptionLabel,
  formatLatentWidth,
  shortHookType,
} from './saeOptionLabel';

describe('shortHookType', () => {
  it('maps training-config hook types', () => {
    expect(shortHookType('residual')).toBe('res');
    expect(shortHookType('mlp')).toBe('mlp');
    expect(shortHookType('attention')).toBe('att');
  });

  it('maps SAELens hook points', () => {
    expect(shortHookType('hook_resid_post')).toBe('res');
    expect(shortHookType('hook_resid_pre')).toBe('res');
    expect(shortHookType('hook_mlp_out')).toBe('mlp');
    expect(shortHookType('hook_attn_out')).toBe('att');
    expect(shortHookType('hook_z')).toBe('att');
  });

  it('maps full hook names like blocks.12.hook_resid_post', () => {
    expect(shortHookType('blocks.12.hook_resid_post')).toBe('res');
    expect(shortHookType('blocks.5.hook_mlp_out')).toBe('mlp');
  });

  it('falls back to a stripped short form for unknown hooks', () => {
    expect(shortHookType('blocks.3.hook_normalized')).toBe('normalized');
    expect(shortHookType('hook_embed')).toBe('embed');
  });

  it('returns null for missing input', () => {
    expect(shortHookType(null)).toBeNull();
    expect(shortHookType(undefined)).toBeNull();
    expect(shortHookType('')).toBeNull();
  });
});

describe('formatLatentWidth', () => {
  it('formats 1024-multiples as k', () => {
    expect(formatLatentWidth(8192)).toBe('8k');
    expect(formatLatentWidth(16384)).toBe('16k');
    expect(formatLatentWidth(32768)).toBe('32k');
    expect(formatLatentWidth(24576)).toBe('24k');
  });

  it('formats 1000-multiples as k', () => {
    expect(formatLatentWidth(16000)).toBe('16k');
  });

  it('renders non-round widths raw', () => {
    expect(formatLatentWidth(16380)).toBe((16380).toLocaleString());
  });

  it('returns null for missing or invalid widths', () => {
    expect(formatLatentWidth(null)).toBeNull();
    expect(formatLatentWidth(undefined)).toBeNull();
    expect(formatLatentWidth(0)).toBeNull();
  });
});

describe('composeSAEOptionLabel', () => {
  it('composes the full label', () => {
    expect(
      composeSAEOptionLabel({
        name: 'gemmascope-res',
        modelName: 'gemma-2-2b',
        layer: 12,
        hookType: 'blocks.12.hook_resid_post',
        width: 16384,
      })
    ).toBe('gemmascope-res — gemma-2-2b · L12 · res · 16k');
  });

  it('omits missing chips without leaving artifacts', () => {
    expect(
      composeSAEOptionLabel({ name: 'my-sae', modelName: 'gpt2', layer: null, width: 24576 })
    ).toBe('my-sae — gpt2 · 24k');
    expect(composeSAEOptionLabel({ name: 'my-sae', layer: 0 })).toBe('my-sae — L0');
  });

  it('returns the bare name when nothing else is known', () => {
    expect(composeSAEOptionLabel({ name: 'extr_20260716' })).toBe('extr_20260716');
  });

  it('never renders "undefined" or "null"', () => {
    const label = composeSAEOptionLabel({
      name: 'sae',
      modelName: undefined,
      layer: undefined,
      hookType: undefined,
      width: undefined,
    });
    expect(label).not.toMatch(/undefined|null/);
  });
});
