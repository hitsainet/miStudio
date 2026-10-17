/**
 * Exchange the value you clicked for one you type.
 *
 * A coordinate swap needs two directions. The ranked list already gives the
 * first one exactly — the row you picked, at the layers it actually reached —
 * but the second one used to come from PIN ORDER: the first pinned token that
 * was not this one. That made the partner a thing you configured somewhere else
 * and then hoped for, when it is half the experiment and the token whose rank
 * gets scored.
 *
 * So: the row supplies the FROM side, and this supplies the TO side, as free
 * text. The server never restricted it — `/jlens/token-check`'s own docstring
 * says "ANY single token has one [a direction] — including tokens the readout
 * never surfaced, which are precisely the interesting swap targets" — the UI
 * did.
 *
 * THE LAYERS ARE SEEDED FROM THE ROW, AND THAT IS THE POINT. Three intervention
 * runs came back null before the first real result, every one of them because
 * of layer choice: the concept was absent from the layers being perturbed, and
 * a random control did exactly as well. The layers where the token actually
 * reached top-k are the layers where there is something to exchange. Read out
 * first, then choose layers — seeding it here makes that the default instead of
 * something you have to already know.
 */

import { useEffect, useMemo, useState } from 'react';
import { Repeat, X } from 'lucide-react';

import { describeSpan } from './aggregate';
import { LayerRangePicker } from './LayerRangePicker';
import { TokenVerdict, useTokenCheck } from './tokenCheck';
import { displayToken } from './utils';

interface SwapComposerProps {
  modelId: string;
  /** The token the reader clicked. Verbatim, never trimmed. */
  from: string;
  /** Absolute layers at which `from` reached top-k. */
  fromLayers: number[];
  /**
   * The layers this lens actually covers.
   *
   * FOR A JACOBIAN CLICK THIS IS THE ARTIFACT'S OWN COVERAGE, not the union
   * across lens types. A partial fit — gemma-4-12b-it is fitted at L42-45 of 48
   * — would otherwise offer the whole stack as though the lens spoke for it.
   */
  lensLayers: number[];
  /** Above this many layers the server records `over_layer_budget`. */
  budget: number;
  busy: boolean;
  onCancel: () => void;
  onSubmit: (target: string, layers: number[]) => void;
}

/** The layers this token actually reached, bounded by what the lens covers. */
export function hitLayers(
  fromLayers: number[],
  lensLayers: number[],
): number[] {
  const covered = new Set(lensLayers);
  const hits = fromLayers.filter((l) => covered.has(l));
  // A token whose hits fall entirely outside the lens is an honest
  // out-of-coverage state the reader can see and correct, not an empty
  // selection that reads as "nothing to do".
  return [...(hits.length ? hits : fromLayers)].sort((a, b) => a - b);
}

/**
 * The seeded range: where the token appeared, capped at the layer budget.
 *
 * CAPPED, AND AT THE DEEP END. A ranked-row click passes every layer the token
 * reached, and a common token reaches all of them — one click used to hook the
 * whole stack at strength 1, which is the oversteer BR-017 v0.2 warns about for
 * small models. The deepest are kept because shallow hits are mostly the junk
 * bands the non-word filter exists to declutter, and the readout that motivated
 * the click is the one nearer the output.
 *
 * This is a DEFAULT, not a clamp: widening the range afterwards is allowed and
 * the over-budget warning says what it costs.
 */
export function seedRange(
  fromLayers: number[],
  lensLayers: number[],
  budget?: number,
): [number, number] | null {
  const hits = hitLayers(fromLayers, lensLayers);
  if (!hits.length) return null;
  const use =
    budget && budget > 0 && hits.length > budget ? hits.slice(-budget) : hits;
  return [Math.min(...use), Math.max(...use)];
}

export function SwapComposer({
  modelId,
  from,
  fromLayers,
  lensLayers,
  budget,
  busy,
  onCancel,
  onSubmit,
}: SwapComposerProps) {
  const [target, setTarget] = useState('');
  const { checks, checking, check, rejected } = useTokenCheck(modelId);

  const bounds = useMemo(() => {
    const all = lensLayers.length ? lensLayers : fromLayers;
    return all.length
      ? ([Math.min(...all), Math.max(...all)] as [number, number])
      : null;
  }, [lensLayers, fromLayers]);

  const [range, setRange] = useState<[number, number] | null>(() =>
    seedRange(fromLayers, lensLayers, budget),
  );
  // Re-seed when the reader clicks a DIFFERENT row without closing first.
  //
  // KEYED ON CONTENT, NOT ARRAY IDENTITY. The parent passes
  // `meta.layers_by_type[type] ?? []`, and when that key is missing the `??`
  // yields a FRESH array every render. Depending on the array itself then
  // re-ran this effect forever — each pass calling `setRange` with a new tuple,
  // which re-rendered, which built another array — and it wiped the typed
  // target on every loop.
  const seedKey = `${from}|${fromLayers.join(',')}|${lensLayers.join(',')}|${budget}`;
  useEffect(() => {
    setRange(seedRange(fromLayers, lensLayers, budget));
    setTarget('');
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on content
  }, [seedKey]);

  /**
   * The layers that will actually be sent, as absolute numbers.
   *
   * DRAWN FROM THE TOKEN'S OWN HITS, not the lens axis. Filtering the whole
   * axis by the range hooked every layer BETWEEN the token's hits as well: a
   * token appearing only at L5 and L20 hooked all sixteen, fourteen of which
   * carry none of the concept. That is precisely the shape that produced three
   * null runs — a matched random control does just as well where there is
   * nothing to perturb.
   */
  const selected = useMemo(() => {
    const pool = hitLayers(fromLayers, lensLayers);
    if (!range) return pool;
    const [lo, hi] = range;
    return pool.filter((l) => l >= lo && l <= hi);
  }, [range, lensLayers, fromLayers]);

  /**
   * The token's hits fall entirely outside what this lens covers.
   *
   * `hitLayers` falls back to the raw hits so the reader can SEE the mismatch
   * rather than facing an empty selection — but submitting it credits an
   * artifact for layers it was never fitted at, which the server now refuses
   * with a 400 AFTER the composer has closed and dropped the typed target.
   * Better to say so here, while the text is still on screen.
   */
  const outOfCoverage =
    lensLayers.length > 0 &&
    fromLayers.length > 0 &&
    !fromLayers.some((l) => lensLayers.includes(l));

  const sameToken = target === from;
  // The reason is rendered ONCE, by TokenVerdict beside the input. Repeating
  // it by the button gave two copies of the same sentence.
  //
  // A MULTI-PIECE TARGET NO LONGER BLOCKS THE RUN. It used to, and that
  // excluded most of the interesting vocabulary — the common case is an
  // ORDINARY WORD the tokenizer splits into subwords, not a phrase. The server
  // takes the MEAN of the pieces' rows and scores the FIRST piece, which is
  // weaker evidence and is said so both here and in the recorded caveat.
  const verdict = rejected(target);
  // `usable` IS FALSE FOR TWO DIFFERENT THINGS, and only one of them is a
  // warning. Dropping the whole rejection check to allow multi-piece targets
  // also allowed n_tokens === 0 — "encodes to nothing" — which has no
  // unembedding row at all and dies in the worker after a 202 and a slot on
  // the single-GPU queue. That is the exact waste `/jlens/token-check` exists
  // to prevent.
  const encodesToNothing = verdict?.n_tokens === 0;
  const multiPiece = Boolean(verdict) && !encodesToNothing;
  const ready =
    Boolean(target.trim()) &&
    !sameToken &&
    !encodesToNothing &&
    !outOfCoverage &&
    selected.length > 0;

  return (
    <div
      className="mb-3 rounded-lg border border-sky-300 bg-sky-50/60 p-3 dark:border-sky-800 dark:bg-sky-950/30"
      data-testid="jlens-swap-composer"
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="flex items-center gap-1.5 text-xs font-medium text-slate-800 dark:text-slate-100">
          <Repeat className="h-3.5 w-3.5 text-sky-600 dark:text-sky-400" />
          Swap a readout value
        </h3>
        <button
          type="button"
          onClick={onCancel}
          className="rounded p-0.5 text-slate-500 hover:bg-slate-200 dark:text-slate-400 dark:hover:bg-slate-700"
          aria-label="Cancel swap"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="mb-2 flex flex-wrap items-center gap-2">
        {/* THE ROW YOU PICKED, not editable. Same rendering as every other
            token in the panel, with the raw string on hover — ' Paris' and
            'Paris' are different directions and must not look identical. */}
        <span
          className="rounded border border-slate-300 bg-white px-1.5 py-0.5 font-mono text-[11px] text-slate-800 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
          title={JSON.stringify(from)}
          data-testid="swap-from"
        >
          {displayToken(from)}
        </span>
        <span className="text-slate-400">→</span>
        <label htmlFor="swap-target" className="sr-only">
          Replacement token
        </label>
        <input
          id="swap-target"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          onBlur={() => void check(target)}
          placeholder=" football"
          spellCheck={false}
          className="w-40 rounded border border-slate-300 bg-white px-1.5 py-0.5 font-mono text-[11px] dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
          data-testid="swap-target"
        />
        <TokenVerdict check={checks[target]} busy={checking} />
      </div>

      {/* THE HAZARD THE REFERENCE UI PUTS IN ITS LABEL. Most words are a
          different token with and without the leading space, and the one you
          want is almost always the spaced one — it is the form that follows a
          word. Saying so costs a line and saves a null run. */}
      <p className="mb-2 text-[10px] text-slate-500 dark:text-slate-400">
        Include the leading space — <code>{'" football"'}</code> and{' '}
        <code>{'"football"'}</code> are different tokens, and the spaced form is
        usually the one that follows a word.
      </p>

      {/* THE CONSEQUENCE ONLY. `TokenVerdict` beside the input already states
          the piece count and the leading-space hint; repeating it here put the
          same sentence on screen twice. */}
      {multiPiece && (
        <p className="mb-2 text-[10px] text-amber-700 dark:text-amber-400">
          This will still run: the direction becomes the mean of those pieces and
          only the first is scored, which is weaker evidence than a single token.
        </p>
      )}
      {outOfCoverage && (
        <p className="mb-2 text-[10px] text-rose-700 dark:text-rose-400">
          This token only appears at layers this lens was not fitted for, so a
          swap here cannot be credited to it. Re-read within the lens's own
          layers, or pick a token it covers.
        </p>
      )}

      {encodesToNothing && (
        <p className="mb-2 text-[10px] text-rose-700 dark:text-rose-400">
          That encodes to no tokens at all, so there is no coordinate to
          exchange with. This one cannot run.
        </p>
      )}

      {sameToken && (
        <p className="mb-2 text-[10px] text-amber-700 dark:text-amber-400">
          A swap exchanges two DIFFERENT coordinates — one token would run an
          additive steer under a swap's name.
        </p>
      )}

      {bounds && (
        <div className="mb-2">
          <LayerRangePicker
            min={bounds[0]}
            max={bounds[1]}
            value={range}
            onChange={setRange}
            hint={
              // SEEDED FROM WHERE THE TOKEN ACTUALLY IS. Perturbing layers that
              // carry none of the concept is what produced three null runs; a
              // matched random control does just as well there.
              `Seeded from the layers ${displayToken(from)} actually reached. ` +
              'Perturbing layers that carry none of the concept reads as no effect.'
            }
          />
        </div>
      )}

      <p
        className="mb-2 text-[10px] text-slate-600 dark:text-slate-300"
        data-testid="swap-layer-summary"
      >
        Will hook {describeSpan(selected)}.
        {selected.length > budget && (
          <span className="text-amber-700 dark:text-amber-400">
            {' '}
            Over the {budget}-layer budget for this model — the run is recorded
            as <code>over_layer_budget</code>, and swaps oversteer easily.
          </span>
        )}
      </p>

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => onSubmit(target, selected)}
          disabled={!ready || busy}
          className="inline-flex items-center gap-1 rounded bg-sky-600 px-2 py-1 text-[11px] font-medium text-white hover:bg-sky-700 disabled:cursor-not-allowed disabled:opacity-50"
          data-testid="swap-run"
        >
          <Repeat className="h-3 w-3" />
          {busy ? 'Running…' : 'Swap'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="rounded border border-slate-300 px-2 py-1 text-[11px] text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
