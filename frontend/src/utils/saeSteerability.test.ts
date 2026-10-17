/**
 * The picker's copy of the backend's SAE rule — its BEHAVIOUR.
 *
 * The rule itself lives in `backend/src/services/sae_hook_support.py`, and the guard that
 * keeps the two copies in step lives beside it, in
 * `backend/tests/unit/test_the_picker_rule_matches_the_backend.py`: Python reads this
 * file's token lists and fails when they and the frozensets diverge. A ported rule that
 * quietly falls behind is worse than no port, because the panel would then offer an SAE the
 * backend refuses (or hide one it accepts) while both look right in isolation.
 *
 * It is asserted from that side rather than this one for two reasons: Vite will not read a
 * file outside its project root (the `?raw` import of the backend rule failed to load, and
 * the whole file silently stopped running), and `tsconfig.test.json` deliberately carries no
 * `@types/node`, so `node:fs` added three errors to a type-check ratchet that is down-only.
 */

import { describe, expect, it } from 'vitest';

import { classifyHook, isSteerable, steeringRefusalReason } from './saeSteerability';

describe('classifyHook', () => {
  it.each([
    ['mlp', 'mlp'],
    ['hook_mlp_out', 'mlp'],
    ['blocks.3.mlp.hook_post', 'mlp'],
    ['transcoder', 'mlp'],
    ['blocks.3.ln2.hook_normalized', 'mlp'],
    ['model.layers.3.feed_forward', 'mlp'],
    ['model.layers.3.ffn_norm', 'mlp'],
  ])('%s is MLP-side', (hook, kind) => expect(classifyHook(hook)).toBe(kind));

  it.each([
    ['attention', 'attention'],
    ['att', 'attention'],
    ['blocks.7.hook_attn_out', 'attention'],
    ['hook_z', 'attention'],
  ])('%s is attention', (hook, kind) => expect(classifyHook(hook)).toBe(kind));

  it.each([
    ['embed_tokens', 'embedding'],
    ['hook_embed', 'embedding'],
    ['embedding', 'embedding'],
  ])('%s is the embedding output', (hook, kind) => expect(classifyHook(hook)).toBe(kind));

  it.each([
    ['blocks.0.hook_resid_pre', 'resid_pre'],
    ['blocks.0.hook_resid_mid', 'resid_mid'],
  ])('%s reads before the layer output', (hook, kind) => expect(classifyHook(hook)).toBe(kind));

  it.each([
    ['residual', 'residual'],
    ['resid_post', 'residual'],
    ['blocks.12.hook_resid_post', 'residual'],
    ['model.layers.17.output', 'residual'],
    // Names that merely CONTAIN the letters of another hook are not that hook: the rule
    // matches name TOKENS, exactly as the backend's does.
    ['hook_resid_post_mlpish', 'residual'],
    ['battery_resid_post', 'residual'],
  ])('%s is the layer output', (hook, kind) => expect(classifyHook(hook)).toBe(kind));

  it.each([null, undefined, ''])('%s is unrecorded, which reads as residual', (hook) => {
    expect(classifyHook(hook)).toBeNull();
    expect(steeringRefusalReason({ hook_type: hook, layer: 3 })).toBeNull();
  });
});

describe('steeringRefusalReason', () => {
  it('accepts a residual SAE at a recorded layer', () => {
    expect(steeringRefusalReason({ hook_type: 'residual', layer: 12 })).toBeNull();
    expect(isSteerable({ hook_type: 'blocks.12.hook_resid_post', layer: 12 })).toBe(true);
  });

  it('accepts a RECORDED layer 0 — only a missing layer is refused', () => {
    expect(steeringRefusalReason({ hook_type: 'residual', layer: 0 })).toBeNull();
  });

  it('refuses an SAE that records no layer, and says how to fix it', () => {
    const reason = steeringRefusalReason({ hook_type: 'residual', layer: null });
    expect(reason).toMatch(/no layer recorded/);
    expect(reason).toMatch(/backfill/);
    expect(steeringRefusalReason({ hook_type: 'residual' })).toBe(reason);
  });

  it.each([
    ['hook_mlp_out', /MLP-side/],
    ['blocks.7.hook_attn_out', /attention-output/],
    ['embed_tokens', /before the first layer/],
    ['blocks.0.hook_resid_pre', /before the layer output/],
    ['blocks.0.hook_resid_mid', /inside the layer/],
  ])('refuses %s with its reason', (hook, matcher) => {
    expect(steeringRefusalReason({ hook_type: hook, layer: 3 })).toMatch(matcher);
  });

  it('names the hook before the layer when both are wrong', () => {
    expect(steeringRefusalReason({ hook_type: 'mlp', layer: null })).toMatch(/MLP-side/);
  });
});
