/**
 * Which layers an artifact actually covers, at a glance.
 *
 * WHY THIS EXISTS. A 9-of-16-layer LFM2 lens was indistinguishable from a full
 * one everywhere in the product — same card, same size line, same Validate
 * button — right up until a readout asked for layer 0 and the transport
 * refused. The coverage was knowable the whole time and simply never shown.
 *
 * An UNKNOWN coverage (empty list) renders as "not recorded" rather than as an
 * empty strip. An empty strip says "this artifact covers nothing", which is a
 * claim the listing never made.
 */

interface LayerCoverageProps {
  /** Layers the artifact holds. Empty = unknown, which is NOT zero coverage. */
  covered: number[];
  /** The model's full layer count, or null when the model's dims are unknown. */
  total: number | null;
  /**
   * The recipe's target block, when recorded.
   *
   * A `penultimate` target makes the TOP LAYER UNFITTABLE — its gradient to the
   * target is zero by causality — so a complete fit covers `total - 1` layers.
   * Without this the strip reports a correct artifact as incomplete and colours
   * it amber, which is a recipe choice rendered as a defect.
   */
  targetLayer?: string | null;
}

export function LayerCoverage({
  covered,
  total,
  targetLayer,
}: LayerCoverageProps) {
  if (!total) {
    return (
      <span className="text-[10px] text-slate-500 dark:text-slate-500">
        layer coverage needs the model's dimensions
      </span>
    );
  }
  if (!covered.length) {
    return (
      <span className="text-[10px] text-amber-600 dark:text-amber-400">
        coverage not recorded — refit to know which layers this holds
      </span>
    );
  }

  const set = new Set(covered);
  // How many layers this recipe CAN cover, which is what "complete" means.
  const fittable = targetLayer === 'penultimate' ? Math.max(total - 1, 1) : total;
  const complete = covered.length >= fittable;

  return (
    <span className="flex items-center gap-1.5">
      <span
        className="flex gap-[1px]"
        role="img"
        aria-label={`covers ${covered.length} of ${fittable} layers`}
      >
        {Array.from({ length: total }, (_, layer) => (
          <span
            key={layer}
            title={`L${layer}${set.has(layer) ? '' : ' — not fitted'}`}
            className={`inline-block h-3 w-[3px] rounded-[1px] ${
              set.has(layer)
                ? 'bg-emerald-500'
                : 'bg-slate-300 dark:bg-slate-700'
            }`}
          />
        ))}
      </span>
      <span
        className={`font-mono text-[10px] ${
          complete
            ? 'text-slate-500 dark:text-slate-500'
            : 'text-amber-600 dark:text-amber-400'
        }`}
      >
        {covered.length}/{fittable} layers
        {fittable !== total ? ' · penultimate target' : ''}
      </span>
    </span>
  );
}

/** Layers of `total` the artifact does NOT hold. */
export function missingLayers(covered: number[], total: number): number[] {
  const set = new Set(covered);
  return Array.from({ length: total }, (_, i) => i).filter((l) => !set.has(l));
}
