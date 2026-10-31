/**
 * Which SAEs steering will actually accept — the backend's rule, ported.
 *
 * The picker offered every READY SAE, so choosing a Gemma Scope MLP or attention SAE, or
 * one that records no layer, produced a 422 only after the user had picked features and
 * pressed Generate (review R3-B: "the steering picker still offers MLP and attention SAEs
 * that steering refuses"). The refusal itself is right and stays where it is — the backend
 * is the authority, and it still answers 422. This is only so the panel does not OFFER
 * what it knows will be refused, and says why.
 *
 * ⚠ TWO COPIES OF ONE RULE. The authority is
 * `backend/src/services/sae_hook_support.py` (`classify_hook`,
 * `non_residual_hook_reason`, `unrecorded_layer_reason`). `saeSteerability.test.ts` reads
 * that file and fails if its token sets and this file's drift apart, so the copy cannot
 * quietly fall behind. Never relax anything here without changing it there first.
 */

/** Tokens that name an MLP-side hook: its output, its input, or a transcoder across it. */
export const MLP_TOKENS = ['mlp', 'transcoder', 'transcoders', 'ffn', 'feedforward'];
/** Tokens that name an attention hook: its output, or a per-head quantity inside it. */
export const ATTENTION_TOKENS = ['attn', 'attention', 'att', 'z', 'q', 'k', 'v', 'result', 'pattern'];
/** Tokens that name the embedding output: the residual stream before the first layer. */
export const EMBEDDING_TOKENS = ['embed', 'embedding', 'embeddings'];
/** Tokens that name the residual stream. */
export const RESIDUAL_TOKENS = ['resid', 'residual', 'res'];

export type HookKind =
  | 'residual'
  | 'mlp'
  | 'attention'
  | 'embedding'
  | 'resid_pre'
  | 'resid_mid'
  | null;

/** Split a hook name into name TOKENS, exactly as the backend's `_tokens` does. */
function tokensOf(hookType: string): string[] {
  return hookType
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 0);
}

/**
 * The kind of point a hook names, by the backend's rule. `null` when unrecorded — which
 * every consumer reads as residual, as it always has.
 */
export function classifyHook(hookType: string | null | undefined): HookKind {
  if (!hookType) return null;
  const tokens = tokensOf(hookType);
  const has = (list: string[]) => tokens.some((token) => list.includes(token));
  const feedForward = tokens.some((token, i) => token === 'feed' && tokens[i + 1] === 'forward');

  if (has(MLP_TOKENS) || (tokens.includes('ln2') && tokens.includes('normalized')) || feedForward) {
    return 'mlp';
  }
  if (has(ATTENTION_TOKENS)) return 'attention';
  if (has(EMBEDDING_TOKENS)) return 'embedding';
  if (has(RESIDUAL_TOKENS)) {
    if (tokens.includes('pre')) return 'resid_pre';
    if (tokens.includes('mid')) return 'resid_mid';
  }
  return 'residual';
}

/**
 * Why steering would refuse this SAE, short enough for an option label, or `null` when it
 * would accept it. The backend's own message (longer, and with the repair instructions) is
 * still what a 422 carries.
 */
export function steeringRefusalReason(
  sae: { hook_type?: string | null; layer?: number | null },
): string | null {
  const kind = classifyHook(sae.hook_type);
  if (kind === 'mlp') return 'MLP-side SAE — steering adds directions at the layer output';
  if (kind === 'attention') return 'attention-output SAE — steering adds directions at the layer output';
  if (kind === 'embedding') return 'embedding SAE — it reads before the first layer';
  if (kind === 'resid_pre') return 'resid_pre SAE — it reads before the layer output';
  if (kind === 'resid_mid') return 'resid_mid SAE — it reads inside the layer';
  // A recorded layer 0 is a real layer; only a missing one is refused.
  if (sae.layer === null || sae.layer === undefined) {
    return 'no layer recorded — steering would use layer 0; run the SAE hook backfill';
  }
  return null;
}

/** Whether steering will accept this SAE. */
export function isSteerable(sae: { hook_type?: string | null; layer?: number | null }): boolean {
  return steeringRefusalReason(sae) === null;
}
