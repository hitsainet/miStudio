/**
 * One probe run's tile: what it ran ON, when, for how long, and what can be done to it.
 *
 * ⚠ THE TILE USED TO SHOW AN ID, A STATUS AND A PERCENTAGE, AND NOTHING ELSE. Five runs against
 * different models and corpora were visually identical apart from twelve hex characters, a
 * finished run looked the same as a live one, and a cancelled or failed run could never be cleared
 * — the DELETE route existed and nothing called it. An operator could not answer "which of these
 * is the LFM2 one", "is this still going", or "how long did it take".
 *
 * What each addition is for:
 *   * **the model and the views** — a run is defined by what it read, and an id is not that
 *   * **started / ended / duration** — "28 min" is the number that decides whether to run another
 *   * **a live pulse and a ticking elapsed** — a stage at 30% for twenty minutes is normal here
 *     (`pooled_capture` is one forward pass over 8,000 rows), so "is it moving" cannot be answered
 *     from the percentage alone
 *   * **delete, behind a confirm** — a run owns a multi-gigabyte token memmap, and deleting it is
 *     the only thing that reclaims that disk
 */
import { useEffect, useState } from 'react';
import { Trash2, Square, Loader2 } from 'lucide-react';
import { PrecisionBadge } from '../common/PrecisionBadge';

import type { ProbeRun } from '../../types/probeMonitor';
import { ProgressBar } from '../common/ProgressBar';

interface RunTileProps {
  run: ProbeRun;
  /** `{id: display name}` for models and probe views, so the tile names them rather than ids. */
  modelNames?: Record<string, string>;
  datasetNames?: Record<string, string>;
  onCancel: (id: string) => void;
  onDelete: (id: string) => void;
  children?: React.ReactNode;
}

const LIVE = new Set(['pending', 'running', 'cancelling']);

/** `1h 23m 51s`, or `null` when there is nothing honest to say. */
export function formatDuration(ms: number | null): string | null {
  if (ms === null || !Number.isFinite(ms) || ms < 0) return null;
  const total = Math.floor(ms / 1000);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours) return `${hours}h ${minutes}m ${seconds}s`;
  if (minutes) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

/**
 * How long a run took, or has been going.
 *
 * ⚠ AN UNFINISHED RUN IS MEASURED AGAINST `now`, NOT LEFT BLANK. "started 40 minutes ago" is the
 * thing an operator actually wants, and a blank duration on a live run reads as "no information"
 * when the information is available.
 */
/**
 * How long a run took, or null when that cannot be said.
 *
 * ⚠ IT REFUSES TO EXTRAPOLATE FOR A RUN THAT HAS ENDED, and that refusal is the whole point.
 * This used to fall back to `now` whenever `completed_at` was missing, which is right for a LIVE
 * run and a fabrication for a terminal one. `pmr_5d81ad3b81f2` failed within minutes, never had
 * `completed_at` stamped (only the success path did that), and the panel displayed
 * "Took 6h 36m 19s" beside "Finished —" — a number that grew on every refresh and that a reader
 * has no way to know is invented.
 *
 * The backend now stamps every terminal path, but the rows already in the database never will be,
 * so this has to hold the line on its own: for a run that is not live and records no end, the
 * honest answer is "—".
 */
export function runDuration(run: ProbeRun, now: number = Date.now()): number | null {
  if (!run.created_at) return null;
  const started = Date.parse(run.created_at);
  if (Number.isNaN(started)) return null;
  if (!run.completed_at) {
    if (!LIVE.has(run.status)) return null;
    return now - started;
  }
  const ended = Date.parse(run.completed_at);
  if (Number.isNaN(ended)) return null;
  return ended - started;
}

function stamp(value: string | null | undefined): string {
  if (!value) return '—';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return '—';
  return parsed.toLocaleString();
}

export function RunTile({
  run,
  modelNames = {},
  datasetNames = {},
  onCancel,
  onDelete,
  children,
}: RunTileProps) {
  const live = LIVE.has(run.status);
  const [confirming, setConfirming] = useState(false);
  // A ticking clock ONLY while the run is live: a finished run's duration is fixed, and
  // re-rendering the whole list every second for nothing is wasteful.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!live) return undefined;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [live]);

  const duration = formatDuration(runDuration(run, now));
  const evalIds = run.eval_dataset_ids ?? [];
  const name = (id: string | null | undefined, table: Record<string, string>) =>
    (id && table[id]) || id || '—';
  const evalNames = evalIds.map((id) => name(id, datasetNames)).join(', ');
  // ⚠ A BAR NEEDS A NUMBER. A live run whose progress has not been reported yet would otherwise
  // draw an empty bar, which reads as "nothing has happened" rather than "nothing has been said".
  // The spinner and the stage carry it until a percentage exists.
  //
  // ⚠ NOT CLAMPED HERE. It was, and a mutation control proved the clamp was dead code:
  // `common/ProgressBar` clamps to 0..100 itself, so removing this one changed no behaviour and no
  // test. Two owners for one rule is how they drift — the component that draws the bar owns the
  // bound, and `test_probe_progress_bar` in ProgressBar.test.tsx is where it is pinned.
  const percent =
    run.progress === null || run.progress === undefined ? null : run.progress;

  return (
    <li
      className={`rounded border px-3 py-2 ${
        live ? 'border-sky-700 bg-sky-950/20' : 'border-slate-700 bg-slate-900/60'
      }`}
      data-testid="run-row"
      data-live={live ? 'true' : 'false'}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="flex items-center gap-2 font-mono text-sm text-slate-200">
            {live ? (
              <Loader2
                className="h-3.5 w-3.5 shrink-0 animate-spin text-sky-400"
                data-testid="run-live-spinner"
                aria-label="running"
              />
            ) : null}
            {run.id}
            <PrecisionBadge label={run.model_dtype_label} />
          </p>
          <p className="text-xs text-slate-400" data-testid="run-status">
            {run.status}
            {/* THE STAGE, not only a percentage: 40% means nothing without
                "layer_selection" beside it. */}
            {run.stage ? ` · ${run.stage}` : ''}
            {run.progress !== null && run.progress !== undefined
              ? ` · ${run.progress.toFixed(0)}%`
              : ''}
            {live && duration ? ` · running ${duration}` : ''}
          </p>
          {run.error_message ? (
            <p className="mt-1 text-xs text-amber-300" data-testid="run-error">
              {run.error_message}
            </p>
          ) : null}
        </div>

        <div className="flex shrink-0 items-center gap-2">
          {live ? (
            <button
              type="button"
              onClick={() => onCancel(run.id)}
              className="flex items-center gap-1 rounded border border-slate-600 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
              data-testid="run-stop"
            >
              <Square className="h-3 w-3" />
              Stop
            </button>
          ) : null}
          {/* ⚠ NOT OFFERED WHILE LIVE. Deleting a running run removes the row its worker reads,
              and the cancel scope treats a missing row as a stop — so it works, but "Stop" is the
              honest button for that intent and this one would hide a cancellation inside a
              delete. */}
          {!live ? (
            confirming ? (
              <span className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => {
                    setConfirming(false);
                    onDelete(run.id);
                  }}
                  className="rounded bg-red-700 px-2 py-1 text-xs text-white"
                  data-testid="run-delete-confirm"
                >
                  Delete
                </button>
                <button
                  type="button"
                  onClick={() => setConfirming(false)}
                  className="rounded border border-slate-600 px-2 py-1 text-xs text-slate-300"
                  data-testid="run-delete-cancel"
                >
                  Keep
                </button>
              </span>
            ) : (
              <button
                type="button"
                onClick={() => setConfirming(true)}
                title="Delete this run, its probes and its artifacts"
                className="flex items-center gap-1 rounded border border-slate-700 px-2 py-1 text-xs text-slate-400 hover:bg-slate-800 hover:text-red-300"
                data-testid="run-delete"
              >
                <Trash2 className="h-3 w-3" />
                Delete
              </button>
            )
          ) : null}
        </div>
      </div>

      {/*
        THE SAME BAR EVERY OTHER RUNNING JOB TILE DRAWS — `common/ProgressBar`, not a sixth
        hand-rolled copy of `h-2 rounded-full overflow-hidden`. Training, tokenization, extraction,
        labeling and Neuronpedia export all show a horizontal bar while they run, and a probe run was
        the only long job reporting its progress as text in a status line.

        The percentage is not repeated here: it is already in the status line above, beside the stage
        that gives it meaning. What the bar adds is the thing text cannot — progress readable without
        being read.
      */}
      {live && percent !== null ? (
        <div className="mt-2 flex items-center gap-2" data-testid="run-progress">
          <ProgressBar progress={percent} showPercentage={false} className="flex-1" />
          {duration ? (
            <span className="shrink-0 font-mono text-[11px] text-slate-400">{duration}</span>
          ) : null}
        </div>
      ) : null}

      {confirming ? (
        <p className="mt-2 rounded border border-red-900 bg-red-950/30 p-2 text-xs text-red-200">
          Deletes the run, every probe trained in it, and its artifact directory — the token
          capture there is gigabytes. This cannot be undone.
        </p>
      ) : null}

      {/*
        ⚠ THE FACTS AND THE LAYER SWEEP SIT SIDE BY SIDE, which is the other half of "use the width".
        The sweep table was stacked UNDER the facts, so a tile spent ~70px on facts and ~110px on a
        narrow table while half of its 1328px sat empty. They share a row from `lg` up and stack below
        it, so a narrow window still reads top to bottom.

        The facts drop to three columns here rather than four: sharing the row leaves them about 55%
        of the tile, and a four-column timestamp wraps. Six facts in three columns is still two rows,
        so nothing is lost.
      */}
      <div className="mt-2 flex flex-col gap-3 lg:flex-row lg:items-start lg:gap-6">
      {/*
        ACROSS THE TILE, NOT DOWN IT. This was a two-column `auto 1fr` list, so six facts became six
        stacked rows and a run tile was taller than the sweep table underneath it — while most of the
        tile's width sat empty. The same six facts now fill two rows on a wide screen and reflow to
        two columns on a narrow one, which is what the space was there for.

        Each cell is label-above-value rather than label-beside-value: inline pairs read as prose and
        stop being scannable once the values differ in length, which these do (a model name beside a
        timestamp beside a comma list of five datasets).
      */}
      <dl
        className="grid min-w-0 grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-3"
        data-testid="run-details"
      >
        <div className="min-w-0">
          <dt className="text-[10px] uppercase tracking-wide text-slate-500">Model</dt>
          <dd className="truncate text-slate-300" title={name(run.model_id, modelNames)}>
            {name(run.model_id, modelNames)}
          </dd>
        </div>

        <div className="min-w-0">
          <dt className="text-[10px] uppercase tracking-wide text-slate-500">Train</dt>
          <dd
            className="truncate text-slate-300"
            title={name(run.train_dataset_id, datasetNames)}
          >
            {name(run.train_dataset_id, datasetNames)}
          </dd>
        </div>

        <div className="min-w-0">
          <dt className="text-[10px] uppercase tracking-wide text-slate-500">Started</dt>
          <dd className="truncate text-slate-300">{stamp(run.created_at)}</dd>
        </div>

        <div className="min-w-0">
          <dt className="text-[10px] uppercase tracking-wide text-slate-500">
            {live ? 'Elapsed' : 'Finished'}
          </dt>
          <dd className="truncate text-slate-300">
            {live ? duration ?? '—' : stamp(run.completed_at)}
          </dd>
        </div>

        {/* Spans, because it holds a comma list of every evaluation view and the others hold one
            value each. `title` carries the full list when the cell truncates. */}
        <div className="col-span-2 min-w-0 sm:col-span-3">
          <dt className="text-[10px] uppercase tracking-wide text-slate-500">Evaluate</dt>
          <dd className="truncate text-slate-300" title={evalNames || undefined}>
            {evalNames || '—'}
          </dd>
        </div>

        {/*
          Shown for every finished run, with an em dash when the duration is unknown. An absent cell
          reads as "this run has no such thing"; "—" reads as "nobody recorded it", which is the
          truth for the runs that failed before `completed_at` was stamped on failure.
        */}
        {!live ? (
          <div className="min-w-0">
            <dt className="text-[10px] uppercase tracking-wide text-slate-500">Took</dt>
            <dd className="truncate text-slate-300" data-testid="run-duration">
              {duration ?? '—'}
            </dd>
          </div>
        ) : null}
      </dl>

        {children ? (
          <div className="min-w-0 shrink-0 lg:max-w-[46%]" data-testid="run-aside">
            {children}
          </div>
        ) : null}
      </div>

    </li>
  );
}
