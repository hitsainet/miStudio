/**
 * SAE dropdown option label composition.
 *
 * SAE names alone often don't distinguish SAEs trained on the same model —
 * different layers, hook types, and latent widths look identical in a plain
 * `<option>` list. This composer appends compact identifying chips:
 *
 *   "gemmascope-res — gemma-2-2b · L12 · res · 16k"
 *
 * Missing fields degrade gracefully (chip omitted, never "undefined").
 * Shared by the Clusters extraction dropdown and the Steering SAE selector.
 */

export interface SAEOptionLabelParts {
  /** Base display name (SAE name, extraction name, or id fallback). */
  name: string;
  /** Model the SAE was trained on, e.g. "gemma-2-2b". */
  modelName?: string | null;
  /** Transformer layer index. */
  layer?: number | null;
  /**
   * Hook type in any of miStudio's forms: "residual" | "mlp" | "attention",
   * SAELens hook points ("hook_resid_post", "hook_mlp_out", "hook_z"), or
   * full hook names ("blocks.12.hook_resid_post").
   */
  hookType?: string | null;
  /** Latent width (d_sae / n_features). */
  width?: number | null;
}

/**
 * Normalize any hook type / hook point spelling to a compact chip:
 * "res" | "mlp" | "att". Unknown hooks fall back to a stripped short form.
 */
export function shortHookType(hookType: string | null | undefined): string | null {
  if (!hookType) return null;
  const h = hookType.toLowerCase();
  if (h.includes('resid')) return 'res'; // residual | hook_resid_pre/post | blocks.N.hook_resid_post
  if (h.includes('mlp')) return 'mlp'; // mlp | hook_mlp_out
  if (h.includes('att') || h.includes('hook_z')) return 'att'; // attention | hook_attn_out | hook_z
  // Unknown hook: strip common prefixes and keep it short.
  const stripped = h.replace(/^blocks\.\d+\./, '').replace(/^hook_/, '');
  return stripped || null;
}

/**
 * Format a latent width compactly: 16384 → "16k", 24000 → "24k",
 * non-round widths render raw with thousands separators.
 */
export function formatLatentWidth(width: number | null | undefined): string | null {
  if (width == null || width <= 0) return null;
  if (width % 1024 === 0) return `${width / 1024}k`;
  if (width % 1000 === 0) return `${width / 1000}k`;
  return width.toLocaleString();
}

/**
 * Compose a dropdown option label: name, then " — " and available chips
 * joined with " · ". With no chips available, returns the bare name.
 */
export function composeSAEOptionLabel(parts: SAEOptionLabelParts): string {
  const chips: string[] = [];
  if (parts.modelName) chips.push(parts.modelName);
  if (parts.layer != null) chips.push(`L${parts.layer}`);
  const hook = shortHookType(parts.hookType);
  if (hook) chips.push(hook);
  const width = formatLatentWidth(parts.width);
  if (width) chips.push(width);
  return chips.length > 0 ? `${parts.name} — ${chips.join(' · ')}` : parts.name;
}
