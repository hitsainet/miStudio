/**
 * Which layers to read out, as an inclusive absolute range.
 *
 * Reading every layer is the expensive default and rarely what is wanted:
 * `check_readout_budget` bounds positions x layers BEFORE capture, so narrowing
 * here makes a long prompt cheaper rather than merely tidier.
 *
 * ABSOLUTE LAYER NUMBERS, never indices into an axis. The axis differs per lens
 * type and per artifact — a partial Jacobian fit covers fewer layers than the
 * logit lens, which needs no artifact at all — so an index means a different
 * layer depending on which lens is being read, and a range that silently moves
 * when the mode changes is worse than no range.
 */

import { useId } from 'react';

interface LayerRangePickerProps {
  /** The full span the model offers, from the readout's own axes. */
  min: number;
  max: number;
  value: [number, number] | null;
  onChange: (next: [number, number] | null) => void;
  /** Re-read with the new range; the range is a REQUEST parameter, not a filter. */
  onApply?: () => void;
  /** A readout is in flight; a second click is a second GPU job. */
  busy?: boolean;
}

export function LayerRangePicker({
  min,
  max,
  value,
  onChange,
  onApply,
  busy = false,
}: LayerRangePickerProps) {
  const loId = useId();
  const hiId = useId();
  const [lo, hi] = value ?? [min, max];
  const isAll = value === null;

  const set = (nextLo: number, nextHi: number) => {
    // CLAMPED AND ORDERED. A range whose ends have crossed selects nothing, and
    // an out-of-range bound would ask the server for a layer the model does not
    // have — refused there, which is correct and a poor way to learn it.
    const a = Math.max(min, Math.min(max, nextLo));
    const b = Math.max(min, Math.min(max, nextHi));
    onChange(a <= b ? [a, b] : [b, a]);
  };

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs" data-testid="jlens-layer-range">
      <span className="text-slate-600 dark:text-slate-400">Layers</span>

      <label htmlFor={loId} className="sr-only">
        First layer
      </label>
      <input
        id={loId}
        type="number"
        min={min}
        max={max}
        value={lo}
        onChange={(e) => set(Number(e.target.value), hi)}
        className="w-16 rounded border border-slate-300 bg-white px-1.5 py-1 font-mono text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
      />
      <span className="text-slate-400">–</span>
      <label htmlFor={hiId} className="sr-only">
        Last layer
      </label>
      <input
        id={hiId}
        type="number"
        min={min}
        max={max}
        value={hi}
        onChange={(e) => set(lo, Number(e.target.value))}
        className="w-16 rounded border border-slate-300 bg-white px-1.5 py-1 font-mono text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
      />

      <span className="text-slate-500 dark:text-slate-500">
        of {min}–{max}
      </span>

      {!isAll && (
        <button
          type="button"
          onClick={() => onChange(null)}
          className="rounded border border-slate-300 px-1.5 py-0.5 text-[11px] text-slate-600 hover:bg-slate-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
        >
          All layers
        </button>
      )}

      {onApply && (
        <button
          type="button"
          onClick={onApply}
          disabled={busy}
          className="rounded disabled:cursor-not-allowed disabled:opacity-50 bg-emerald-600 px-2 py-0.5 text-[11px] font-medium text-white hover:bg-emerald-700"
        >
          {busy ? 'Reading…' : 'Re-read'}
        </button>
      )}

      {/* THE RANGE IS A REQUEST PARAMETER, and saying so prevents the obvious
          misreading: the numbers on screen came from whatever range was read,
          and changing these without re-reading narrows the ranked lists over
          data that was captured under the old range. */}
      <span className="basis-full text-[10px] text-slate-500 dark:text-slate-500">
        Narrowing filters the ranked lists only — the grid, rail and trajectory
        keep showing every layer that was read. Re-read to capture only these.
      </span>
    </div>
  );
}
