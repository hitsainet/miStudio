/**
 * Shared rendering helpers for the J-Lens panel.
 *
 * Nothing here encodes a model property. In particular there is NO band
 * constant: the reference implementation's `BAND = { workspaceStart: 40,
 * motorStart: 90 }` are the source paper's Sonnet-4.5 figures, and BR-002
 * requires the product make porting them impossible by construction. Band
 * geometry arrives as a BandReport or not at all.
 */

/** Make whitespace visible without changing the token's identity. */
export function displayToken(token: string): string {
  return token
    .replace(/^ /, '·')
    .replace(/\n/g, '⏎')
    .replace(/\t/g, '→');
}

/**
 * Emerald ramp keyed to rank, scaled by the top-n the SERVER actually sent.
 *
 * `topN` is a required parameter rather than a module constant on purpose: the
 * reference implementation hardcodes 8, which mis-scales the whole heatmap the
 * moment a readout comes back with a different top-n — legibly, and wrongly.
 */
export function rankColor(rank: number | null, topN: number): string {
  if (rank == null) return 'transparent';
  const alpha = Math.max(0.06, 1 - Math.log(rank) / Math.log(Math.max(topN, 2) + 6));
  return `rgba(52, 211, 153, ${alpha.toFixed(3)})`;
}

/**
 * Top-1 probability below which a readout is DIFFUSE — the top token carries so
 * little of the distribution that reading it as content is unsafe.
 *
 * This is a statement about the readout's own distribution, measured per cell.
 * It is explicitly NOT a layer boundary and asserts nothing about which layers
 * are sensory, workspace or motor; those come from a band report or not at all.
 * Diffuse readouts cluster in early layers, which is exactly the expectation
 * FPRD §3.7 requires the panel to surface rather than hide.
 */
export const DIFFUSE_TOP1_PROB = 0.1;

export function isDiffuse(topProb: number | undefined): boolean {
  return topProb === undefined || topProb < DIFFUSE_TOP1_PROB;
}

/** Stable colours for pinned-token series, shared by the grid legend and chart. */
/**
 * One colour per pin, and there must be AT LEAST `MAX_PINNED` of them.
 *
 * Both consumers index with `% PIN_COLORS.length`, so a palette shorter than
 * the pin cap does not crash — it silently RECYCLES, and two different pinned
 * tokens get the same colour in the trajectory chart and the chip list. That is
 * a misread rather than an error: the user compares two lines believing they
 * are one token. `utils.test.ts` pins the relationship so raising the cap
 * cannot quietly start recycling again.
 *
 * ORDERED FOR ADJACENT CONTRAST, not by hue. Pins are assigned in the order
 * they are made, so what matters is that index 0 and index 1 look nothing
 * alike — a palette sorted around the colour wheel gives neighbouring pins
 * neighbouring hues, which is the one arrangement guaranteed to be hard to
 * read. Warm and cool alternate instead.
 *
 * All sixteen sit in the Tailwind 400-500 band so they stay legible against
 * both the light and dark themes; a darker set would read well on white and
 * disappear as a chart stroke on slate-900. Six spare beyond `MAX_PINNED`
 * leave room to raise the cap without revisiting this.
 */
export const PIN_COLORS = [
  '#34d399', // emerald
  '#f472b6', // pink
  '#60a5fa', // blue
  '#f59e0b', // amber
  '#a78bfa', // violet
  '#a3e635', // lime
  '#f87171', // red
  '#22d3ee', // cyan
  '#fb923c', // orange
  '#818cf8', // indigo
  '#4ade80', // green
  '#e879f9', // fuchsia
  '#2dd4bf', // teal
  '#fbbf24', // yellow
  '#38bdf8', // sky
  '#c084fc', // purple
];


/**
 * Cell colour for DIFF, keyed on RANK DISPLACEMENT rather than agreement.
 *
 * `rank` is where the Jacobian lens's top token sits in the LOGIT lens's list
 * at the same layer: 0 means both lenses lead with it, and `null` means the
 * logit lens does not carry it in its top-n at all.
 *
 * Agreement alone was one bit, and it hid the quantity worth seeing. A cell
 * where the logit lens ranks the same token second is nearly agreement; one
 * where it does not rank it at all is the Jacobian lens seeing something the
 * logit lens cannot, which is the entire reason the substrate exists.
 */
export function diffColor(rank: number | null, topN: number): string {
  // ONE-BASED, matching `rankOf` (MIS-E2E-129).
  //
  // This was written for a 0-based rank while `rankOf` returns `i + 1` and
  // therefore NEVER returns 0. Three consequences, all visible:
  //
  //   * rank 1 — the two lenses agreeing on the top token, the thing the Diff
  //     view exists to locate — fell through to the amber ramp and was shaded
  //     as DISAGREEMENT;
  //   * the "same top token" branch below was unreachable, so the legend
  //     advertised a swatch no cell could ever receive (the cheap tell that an
  //     index base is wrong);
  //   * the shading contradicted the `first diverge at L…` badge beside it,
  //     which `firstDisagreement` computes correctly.
  //
  // `rankColor` cannot move to 0-based — it takes `Math.log(rank)`, so rank 0
  // would be -Infinity — and "rank 1 = best" is what the word means. So the
  // base is 1 everywhere and this function moved, not `rankOf`.
  if (rank === 1) return 'rgba(100,116,139,.16)';   // both lenses lead with it
  if (rank === null) return 'rgba(244,63,94,.34)';  // outside the top N entirely
  // Ramp amber -> red across the visible depth, starting at rank 2. Guard
  // topN <= 1 so a top-1 readout does not divide by zero and paint every cell
  // the same.
  const span = Math.max(topN - 1, 1);
  const t = Math.min((rank - 1) / span, 1);
  return `rgba(245,158,11,${(0.18 + 0.16 * t).toFixed(3)})`;
}
