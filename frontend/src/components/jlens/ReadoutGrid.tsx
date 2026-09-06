/**
 * Position x layer readout grid.
 *
 * THE LAYER AXIS COMES FROM THE STREAM. `axis` is `meta.layers_by_type[type]`;
 * this component has no idea how many layers a model has and must not acquire
 * one. The reference implementation hardcodes 21 layers at 0,5,...,100, which
 * renders a complete, plausible and entirely wrong grid on a 16- or 26-layer
 * model.
 *
 * `axis[i]` is the ABSOLUTE layer number for display; `i` is the index into the
 * slice rows. Those two are only interchangeable when the axis happens to be
 * 0..n-1, so every lookup below uses `i` and every label uses `axis[i]`.
 *
 * BANDS ARE DATA. With no BandReport the grid draws no shading and says why —
 * there is no default band object anywhere in this feature (BR-002).
 */

import { useMemo } from 'react';
import { rankColor, diffColor, displayToken, isDiffuse } from './utils';
import { rankOf } from '../../stores/jlensStore';
import type {
  BandReport,
  LensMode,
  LensTokenMessage,
  LensTypeSlice,
} from '../../types/jlens';

type BandName = 'sensory' | 'workspace' | 'motor';

const BAND_LABEL: Record<BandName, string> = {
  sensory: 'Sensory',
  workspace: 'Workspace',
  motor: 'Motor',
};

/** Band membership of an absolute layer, only ever from a report. */
function bandOf(layer: number, report: BandReport | null): BandName | null {
  if (!report) return null;
  if (layer < report.workspace_start) return 'sensory';
  if (layer < report.motor_start) return 'workspace';
  return 'motor';
}

interface ReadoutGridProps {
  axis: number[];
  tokens: LensTokenMessage[];
  topN: number;
  mode: LensMode;
  /** Slice lookup for one token, already bound to the mode's read type. */
  sliceOf: (token: LensTokenMessage) => LensTypeSlice | undefined;
  /** Logit slice, used only by DIFF to compare against the Jacobian. */
  logitSliceOf: (token: LensTokenMessage) => LensTypeSlice | undefined;
  /**
   * The LOGIT lens's own layer axis.
   *
   * Required because the two lenses no longer share one: a Jacobian artifact
   * covers the layers it was fitted for, and `layers_by_type` is per type
   * precisely so they can differ. Comparing row i of one against row i of the
   * other silently compares different layers the moment they do.
   */
  logitAxis: number[];
  /** Layers where J is the identity, so Diff is empty BY CONSTRUCTION. */
  degenerateLayers?: number[];
  pinned: string[];
  selPos: number;
  selLayerIdx: number;
  bandReport: BandReport | null;
  onSelect: (pos: number, layerIdx: number) => void;
  onHover: (h: { pos: number; layerIdx: number } | null) => void;
}

export function ReadoutGrid({
  axis,
  tokens,
  topN,
  mode,
  sliceOf,
  logitSliceOf,
  logitAxis,
  degenerateLayers = [],
  pinned,
  selPos,
  selLayerIdx,
  bandReport,
  onSelect,
  onHover,
}: ReadoutGridProps) {
  // NO EARLY RETURN BEFORE THE HOOKS (MIS-E2E-023).
  //
  // The empty-readout guard used to sit here, above two `useMemo` calls, so on
  // a render where `axis` or `tokens` was empty React saw a SHORTER hook list
  // than on the previous render — "rendered fewer hooks than expected", which
  // unmounts the tree. Two `react-hooks/rules-of-hooks` errors, unreported
  // because lint does not run in CI (MIS-E2E-024).
  //
  // It was assessed as unreachable in practice, because the backend emits
  // `types` and `layers_by_type` from the same tuple so the axis is never empty
  // when tokens are not. That makes it a latent crash rather than a live one —
  // and "currently unreachable" is a property of today's backend, not of this
  // component. Moving the guard below the hooks costs nothing and removes the
  // dependency on that coincidence.
  //
  // Descending layer order: the output end of the stack reads at the top, which
  // is how the trajectory is described everywhere else in the product.
  const rows = axis.map((layer, i) => ({ layer, i })).reverse();

  /**
   * The lowest layer at which the two lenses stop agreeing, at the SELECTED
   * position. That crossing is the quantity the Diff view exists to show — it
   * is where the Jacobian lens starts seeing something the logit lens does not
   * — and it was previously only findable by scanning the column by eye.
   *
   * Null when the lenses agree everywhere, when there is no logit slice to
   * compare against, or when the mode is not DIFF. Null renders nothing rather
   * than a layer number, because "they never disagree" and "we could not tell"
   * must not look the same.
   */
  const firstDisagreement = useMemo(() => {
    if (mode !== 'DIFF') return null;
    const token = tokens.find((t) => t.position === selPos);
    if (!token) return null;
    const jac = sliceOf(token);
    const logit = logitSliceOf(token);
    if (!jac || !logit) return null;
    const logitRow = new Map<number, number>();
    logitAxis.forEach((l, idx) => logitRow.set(l, idx));

    for (let i = 0; i < axis.length; i += 1) {
      const layer = axis[i];
      const lr = logitRow.get(layer);
      if (lr === undefined) continue;
      const mine = jac.top_tokens[i]?.[0];
      const theirs = logit.top_tokens[lr]?.[0];
      if (mine !== undefined && theirs !== undefined && mine !== theirs) {
        return layer;
      }
    }
    return null;
  }, [mode, tokens, selPos, sliceOf, logitSliceOf, axis, logitAxis]);

  // ABSOLUTE LAYER -> the logit lens's own row index. The two lenses have
  // independent axes, so a Jacobian row number is not a logit row number.
  const logitRowOf = useMemo(() => {
    const map = new Map<number, number>();
    logitAxis.forEach((layer, i) => map.set(layer, i));
    return map;
  }, [logitAxis]);

  // Every hook above runs unconditionally on every render; only now is it safe
  // to return early.
  if (axis.length === 0 || tokens.length === 0) {
    return (
      <p className="text-xs text-slate-500 dark:text-slate-500">
        This readout carries no layers for the selected lens.
      </p>
    );
  }

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs font-medium text-slate-600 dark:text-slate-400">
          {pinned.length
            ? 'Rank of pinned tokens · layer × position'
            : mode === 'DIFF'
              ? 'Jacobian top token, shaded by its rank in the logit lens'
              : 'Top readout · layer × position'}
        </div>
        {mode === 'DIFF' && firstDisagreement !== null && (
          <span className="rounded border border-amber-400 px-1.5 py-0.5 font-mono text-[10px] text-amber-700 dark:border-amber-600 dark:text-amber-400">
            lenses first diverge at L{firstDisagreement}
          </span>
        )}
        {mode === 'DIFF' && !pinned.length && (
          // The ramp is meaningless without saying what it measures, and
          // "disagrees" was never the interesting quantity — HOW FAR the two
          // lenses disagree is.
          <div className="flex items-center gap-3 text-[10px] text-slate-500 dark:text-slate-500">
            <span className="flex items-center gap-1">
              <span
                className="inline-block h-2 w-2 rounded-sm"
                style={{ background: diffColor(1, topN) }}
              />
              same top token
            </span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block h-2 w-2 rounded-sm"
                style={{ background: diffColor(topN, topN) }}
              />
              ranked lower by the logit lens
            </span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block h-2 w-2 rounded-sm"
                style={{ background: diffColor(null, topN) }}
              />
              not in the logit lens's top {topN}
            </span>
          </div>
        )}
        <div className="flex items-center gap-3 text-[10px] text-slate-500 dark:text-slate-500">
          {bandReport ? (
            (['sensory', 'workspace', 'motor'] as BandName[]).map((b) => (
              <span key={b} className="flex items-center gap-1">
                <span
                  className={`inline-block h-2 w-2 rounded-sm ${
                    b === 'workspace'
                      ? 'bg-emerald-500'
                      : b === 'motor'
                        ? 'bg-amber-500'
                        : 'bg-slate-400 dark:bg-slate-600'
                  }`}
                />
                {BAND_LABEL[b]}
              </span>
            ))
          ) : (
            <span data-testid="jlens-no-bands">
              No band report for this model — bands are not shown. Boundaries
              from another model do not transfer.
            </span>
          )}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-separate border-spacing-0">
          <tbody>
            {rows.map(({ layer, i }) => {
              const band = bandOf(layer, bandReport);
              return (
                <tr key={layer}>
                  <td
                    onClick={() => onSelect(selPos, i)}
                    className={`sticky left-0 z-10 cursor-pointer border-r bg-white pr-2 text-right font-mono text-[10px] dark:bg-slate-800 ${
                      selLayerIdx === i
                        ? 'border-emerald-500 text-emerald-600 dark:text-emerald-300'
                        : band === 'workspace'
                          ? 'border-slate-300 text-slate-600 dark:border-slate-600 dark:text-slate-400'
                          : 'border-slate-200 text-slate-400 dark:border-slate-700 dark:text-slate-600'
                    }`}
                  >
                    <span className="flex items-center justify-end gap-1.5">
                      L{layer}
                      <span
                        // The gutter. `transparent` when no band report exists,
                        // which is the honest rendering — BR-002 forbids a
                        // default, so an unbanded model shows no stripe rather
                        // than a neutral one implying "sensory".
                        title={band ? BAND_LABEL[band] : undefined}
                        className={`inline-block h-full min-h-[14px] w-[3px] rounded-[1px] ${
                          band === 'workspace'
                            ? 'bg-emerald-500'
                            : band === 'motor'
                              ? 'bg-amber-500'
                              : band === 'sensory'
                                ? 'bg-slate-400 dark:bg-slate-600'
                                : 'bg-transparent'
                        }`}
                      />
                    </span>
                  </td>
                  {tokens.map((tk) => {
                    const slice = sliceOf(tk);
                    const row = slice?.top_tokens[i];
                    const probs = slice?.top_probs[i];

                    let cellTok = row?.[0] ?? '';
                    let background = 'transparent';
                    let dim = isDiffuse(probs?.[0]);
                    let diffNote = '';
                    const degenerate = degenerateLayers.includes(layer);

                    if (pinned.length) {
                      // Heatmap over the pinned tokens: best (lowest) rank wins
                      // the cell, and an unpinned-but-present token is not shown
                      // at all — the grid is answering "where are MY tokens".
                      let best: number | null = null;
                      let which: string | null = null;
                      for (const p of pinned) {
                        const r = rankOf(slice, i, p);
                        if (r != null && (best == null || r < best)) {
                          best = r;
                          which = p;
                        }
                      }
                      background = rankColor(best, topN);
                      cellTok = which ?? '';
                      dim = best == null;
                    } else if (mode === 'DIFF') {
                      // BY ABSOLUTE LAYER, not row index — the axes differ.
                      // A layer the logit lens does not carry is left blank
                      // rather than compared against row i of something else.
                      const logitRow = logitRowOf.get(layer);
                      const logitSlice =
                        logitRow === undefined ? undefined : logitSliceOf(tk);
                      const mine = row?.[0];
                      if (
                        logitRow === undefined ||
                        logitSlice === undefined ||
                        mine === undefined
                      ) {
                        background = 'transparent';
                      } else {
                        // RANK DISPLACEMENT, not bare agreement. "Disagrees" is
                        // one bit: a cell where the logit lens ranks this token
                        // second and one where it does not rank it at all looked
                        // identical, and the second is the interesting one.
                        const r = rankOf(logitSlice, logitRow, mine);
                        background = degenerate
                          ? // Hatch-free neutral: agreement here is not evidence.
                            'rgba(148,163,184,.10)'
                          : diffColor(r, topN);
                        // WHICH LENS PRODUCED THE TEXT. The cell shows the
                        // JACOBIAN's top token in Diff mode, and nothing said
                        // so — a reader had no way to know which of the two
                        // lenses they were looking at.
                        diffNote = degenerate
                          ? ' · J = I at this layer — the two lenses are the same lens here, so agreement is not a finding'
                          : r === null
                            ? ` · Jacobian: ${mine} — outside the logit lens's top ${topN}`
                            : r === 1
                              ? ` · both lenses lead with ${mine}`
                              // `rankOf` is 1-based, so `#${r}` IS the rank.
                              // `#${r + 1}` reported every rank one too high.
                              : ` · Jacobian: ${mine} — logit ranks it #${r}`;
                      }
                    } else {
                      // BANDS NO LONGER TINT THE CELL. Band shading and the pin
                      // heatmap were competing for the same channel — the one a
                      // reader uses to answer "where is my token" — so a banded
                      // grid made ranks harder to read for information already
                      // carried by the row. Bands moved to the gutter below.
                      background = 'rgba(100,116,139,.10)';
                    }

                    const isSel = selPos === tk.position && selLayerIdx === i;
                    return (
                      <td
                        key={tk.position}
                        onMouseEnter={() => onHover({ pos: tk.position, layerIdx: i })}
                        onMouseLeave={() => onHover(null)}
                        onClick={() => onSelect(tk.position, i)}
                        style={{ background }}
                        title={`L${layer} · pos ${tk.position}${
                          dim ? ' · diffuse readout' : ''
                        }${diffNote}`}
                        className={`cursor-pointer border-b border-r px-1 py-[3px] text-center font-mono text-[10px] leading-tight ${
                          isSel
                            ? 'border-emerald-400'
                            : 'border-slate-100 dark:border-slate-800'
                        } ${
                          dim
                            ? // A DIFFUSE READOUT MUST STILL BE READABLE.
                              // `dark:text-slate-700` was near-black on a
                              // slate cell and invisible on the red DIFF
                              // shading — the cells carrying the MOST
                              // interesting signal (a token the logit lens
                              // does not rank at all) were the hardest to
                              // read. Pink says "low confidence" by being a
                              // different hue rather than by being dim, so
                              // legibility no longer trades against meaning.
                              'text-pink-600 dark:text-pink-300'
                            : 'text-slate-800 dark:text-slate-200'
                        }`}
                      >
                        <span className="block max-w-[64px] truncate">
                          {displayToken(cellTok)}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
            <tr>
              <td className="sticky left-0 bg-white dark:bg-slate-800" />
              {tokens.map((tk) => (
                <td
                  key={tk.position}
                  className="max-w-[64px] truncate px-1 pt-1 text-center font-mono text-[10px] text-slate-500 dark:text-slate-500"
                >
                  {displayToken(tk.token)}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
