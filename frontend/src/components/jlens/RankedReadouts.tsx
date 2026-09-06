/**
 * The two lenses as ranked lists, side by side.
 *
 * The grid answers "what is at this layer and position"; this answers "what is
 * this model poised to say across the stack, and where do the two lenses
 * disagree". Reading the second question off the first means collapsing four
 * dimensions by eye, twice, and holding both results in your head.
 *
 * A COUNT IS MEANINGLESS WITHOUT ITS DENOMINATOR. The same token scores
 * differently under a different layer range and a different top-n, so the range
 * and depth are stated beside the numbers rather than tucked into a tooltip —
 * a bare "130" invites a comparison against another readout that is not valid.
 */

import { useMemo } from 'react';
import { Zap, Repeat } from 'lucide-react';

import type { LensTokenMessage } from '../../types/jlens';
import {
  aggregate,
  describeSpan,
  layersInRange,
  type RankedToken,
} from './aggregate';
import { displayToken } from './utils';

interface RankedReadoutsProps {
  tokens: LensTokenMessage[];
  /** Absolute layer axis per lens type, from `meta.layers_by_type`. */
  axes: Record<string, number[]>;
  types: string[];
  topN: number;
  range: [number, number] | null;
  hideNonWords: boolean;
  onToggleNonWords: (next: boolean) => void;
  /**
   * Steer along this token.
   *
   * The COLUMN's lens type travels with it. The two columns are two different
   * lenses, and a token surfaced by the logit lens — which needs no artifact at
   * all — was crediting the Jacobian artifact for a finding it played no part
   * in, writing an `evidence_rung: 2` record into its `interventions.json`
   * under `lens_type: JACOBIAN_LENS`.
   */
  onSteer?: (token: string, layers: number[], type: string) => void;
  /** Exchange this token's coordinate with another. Carries its column too. */
  onSwap?: (token: string, layers: number[], type: string) => void;
  /**
   * Why swap is unavailable FOR THIS TOKEN, or undefined when it is available.
   *
   * PER TOKEN, not per column. Availability depends on there being a DIFFERENT
   * token to exchange with, so a single column-wide reason gets it wrong for
   * exactly one row: with one token pinned, every other row can swap and the
   * pinned row itself cannot. A column-wide guard let that row queue a request
   * with no partner, which the panel reported as "Swap queued" and the GPU
   * refused seconds later.
   */
  swapDisabledFor?: (token: string) => string | undefined;
  /**
   * WHO THIS TOKEN WOULD BE SWAPPED WITH.
   *
   * NAMED, not implied. The partner is half the experiment — it supplies the
   * second coordinate and it is the token whose RANK gets scored — and it was
   * chosen silently from pin order and never shown. Re-pinning in a different
   * order ran a different experiment under an identical-looking click, and
   * `interventions.json` recorded a `target_token` the user never saw.
   */
  swapPartnerFor?: (token: string) => string | undefined;
}

const LABELS: Record<string, string> = {
  JACOBIAN_LENS: 'Jacobian',
  LOGIT_LENS: 'Logit',
};

/**
 * Which layers a token reached, as a strip.
 *
 * Same idiom as the artifact coverage strip: one bar per layer on the axis, lit
 * where the token appeared. A count says how often; this says WHERE, and the
 * two together distinguish a token that is everywhere from one that spikes.
 */
function LayerStrip({ layers, axis }: { layers: number[]; axis: number[] }) {
  const hit = new Set(layers);
  return (
    <span
      className="inline-flex items-center gap-[1px]"
      role="img"
      aria-label={`present at ${layers.length} of ${axis.length} layers`}
    >
      {axis.map((l) => (
        <span
          key={l}
          title={`L${l}${hit.has(l) ? '' : ' — absent'}`}
          className={`h-3 w-[3px] rounded-[1px] ${
            hit.has(l)
              ? 'bg-emerald-500 dark:bg-emerald-400'
              : 'bg-slate-200 dark:bg-slate-700'
          }`}
        />
      ))}
    </span>
  );
}

/**
 * How many rows a column renders before it stops.
 *
 * STATED, NEVER SILENT. Each row draws one bar per in-range layer, so an
 * uncapped list is O(distinct tokens x layers) DOM nodes — six figures on an
 * ordinary 26-layer readout, rebuilt on every range or filter change. A cap is
 * the right answer; a cap the reader cannot see is how a ranked list starts
 * implying it showed everything.
 */
const MAX_ROWS = 60;

function Column({
  type,
  rows,
  axis,
  topN,
  hidden,
  onSteer,
  onSwap,
  swapDisabledFor,
  swapPartnerFor,
}: {
  type: string;
  rows: RankedToken[];
  axis: number[];
  topN: number;
  hidden: number;
  onSteer?: (token: string, layers: number[], type: string) => void;
  onSwap?: (token: string, layers: number[], type: string) => void;
  swapDisabledFor?: (token: string) => string | undefined;
  swapPartnerFor?: (token: string) => string | undefined;
}) {
  return (
    <section
      className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800"
      data-testid={`ranked-${type}`}
    >
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h3 className="text-xs font-medium text-slate-700 dark:text-slate-200">
          {LABELS[type] ?? type}
        </h3>
        {hidden > 0 && (
          <span className="text-[10px] text-slate-500 dark:text-slate-400">
            {hidden} non-word{hidden === 1 ? '' : 's'} hidden
          </span>
        )}
      </div>
      {/* THE DENOMINATOR, PER COLUMN. Layer axes are per lens type — a partial
          Jacobian artifact covers fewer layers than the logit lens, which needs
          no artifact at all — so one shared span would misstate the second
          column's counts. */}
      <p
        className="mb-1.5 text-[10px] text-slate-500 dark:text-slate-400"
        data-testid={`ranked-span-${type}`}
      >
        over {describeSpan(axis)}, top-{topN}
      </p>

      {rows.length === 0 ? (
        <p className="text-[11px] text-slate-500 dark:text-slate-400">
          Nothing to rank in this range.
        </p>
      ) : (
        <ul className="max-h-80 space-y-[2px] overflow-y-auto">
          {rows.slice(0, MAX_ROWS).map((r) => {
            const swapBlocked = swapDisabledFor?.(r.token);
            const partner = swapPartnerFor?.(r.token);
            return (
            <li
              key={r.token}
              className="group flex items-center gap-2 rounded px-1 py-[2px] hover:bg-slate-50 dark:hover:bg-slate-700/40"
            >
              <span className="w-8 shrink-0 text-right font-mono text-[10px] tabular-nums text-slate-500 dark:text-slate-400">
                {r.count}
              </span>
              {/* SAME RENDERING AS THE REST OF THE PANEL. The grid, the rail
                  and the pin chips all map a leading space to `·`; this column
                  did not, so ' Paris' and 'Paris' — different unembedding rows,
                  different counts — appeared as two adjacent identical rows,
                  in a component whose own header insists a count is meaningless
                  without its denominator. Clicking Steer on one of them steered
                  along a direction the user could not distinguish from the
                  other. `title` carries the raw token for anyone who needs it. */}
              <span
                className="min-w-0 flex-1 truncate font-mono text-[11px] text-slate-800 dark:text-slate-100"
                title={JSON.stringify(r.token)}
              >
                {displayToken(r.token)}
              </span>
              <LayerStrip layers={r.layers} axis={axis} />
              {/* REVEALED ON HOVER, not hidden behind a separate form. The
                  token, its layers, the model and the prompt are all already on
                  screen; making the user re-type them into a card elsewhere is
                  how the intervention surface ended up sending an empty prompt
                  and the entire layer axis. */}
              <span className="flex shrink-0 gap-1 opacity-0 transition group-hover:opacity-100 focus-within:opacity-100">
                <button
                  type="button"
                  onClick={() => onSteer?.(r.token, r.layers, type)}
                  title={`Steer along ${r.token}`}
                  className="inline-flex items-center gap-0.5 rounded border border-emerald-300 px-1 py-[1px] text-[9px] text-emerald-700 hover:bg-emerald-50 dark:border-emerald-700 dark:text-emerald-300 dark:hover:bg-emerald-900/30"
                >
                  <Zap className="h-2.5 w-2.5" />
                  Steer
                </button>
                <button
                  type="button"
                  onClick={() => onSwap?.(r.token, r.layers, type)}
                  disabled={Boolean(swapBlocked)}
                  title={
                    swapBlocked ??
                    (partner
                      ? `Swap ${displayToken(r.token)} with ${displayToken(partner)} — ` +
                        `${displayToken(partner)} is the token whose rank is scored`
                      : `Swap ${displayToken(r.token)} with a pinned token`)
                  }
                  className="inline-flex items-center gap-0.5 rounded border border-sky-300 px-1 py-[1px] text-[9px] text-sky-700 hover:bg-sky-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-sky-700 dark:text-sky-300 dark:hover:bg-sky-900/30"
                >
                  <Repeat className="h-2.5 w-2.5" />
                  {partner ? `Swap ↔ ${displayToken(partner)}` : 'Swap'}
                </button>
              </span>
            </li>
            );
          })}
        </ul>
      )}
      {rows.length > MAX_ROWS && (
        <p className="mt-1 text-[10px] text-slate-500 dark:text-slate-400">
          Showing the top {MAX_ROWS} of {rows.length} tokens.
        </p>
      )}
    </section>
  );
}

export function RankedReadouts({
  tokens,
  axes,
  types,
  topN,
  range,
  hideNonWords,
  onToggleNonWords,
  onSteer,
  onSwap,
  swapDisabledFor,
  swapPartnerFor,
}: RankedReadoutsProps) {
  const columns = useMemo(
    () =>
      types.map((type) => {
        const axis = axes[type] ?? [];
        return {
          type,
          axis: layersInRange(axis, range),
          ...aggregate(tokens, type, axis, range, { hideNonWords }),
        };
      }),
    [tokens, axes, types, range, hideNonWords],
  );

  if (!columns.length) return null;

  return (
    <div className="mb-4" data-testid="jlens-ranked">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <p className="text-[11px] text-slate-600 dark:text-slate-300">
          Ranked by how many{' '}
          <span className="font-medium">(layer, position)</span> cells each token
          reached. Each column states its own range.
        </p>
        <label className="flex items-center gap-1.5 text-[11px] text-slate-600 dark:text-slate-300">
          <input
            type="checkbox"
            checked={hideNonWords}
            onChange={(e) => onToggleNonWords(e.target.checked)}
            className="h-3 w-3"
          />
          Hide non-words
        </label>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {columns.map((c) => (
          <Column
            key={c.type}
            type={c.type}
            rows={c.ranked}
            axis={c.axis}
            topN={topN}
            hidden={c.hiddenNonWords}
            onSteer={onSteer}
            onSwap={onSwap}
            swapDisabledFor={swapDisabledFor}
            swapPartnerFor={swapPartnerFor}
          />
        ))}
      </div>
    </div>
  );
}
