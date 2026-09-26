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

import type { ProbeRun } from '../../types/probeMonitor';

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
export function runDuration(run: ProbeRun, now: number = Date.now()): number | null {
  if (!run.created_at) return null;
  const started = Date.parse(run.created_at);
  if (Number.isNaN(started)) return null;
  const ended = run.completed_at ? Date.parse(run.completed_at) : now;
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

  return (
    <li
      className={`rounded border p-3 ${
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

      {confirming ? (
        <p className="mt-2 rounded border border-red-900 bg-red-950/30 p-2 text-xs text-red-200">
          Deletes the run, every probe trained in it, and its artifact directory — the token
          capture there is gigabytes. This cannot be undone.
        </p>
      ) : null}

      <dl
        className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs"
        data-testid="run-details"
      >
        <dt className="text-slate-500">Model</dt>
        <dd className="truncate text-slate-300">{name(run.model_id, modelNames)}</dd>

        <dt className="text-slate-500">Train</dt>
        <dd className="truncate text-slate-300">{name(run.train_dataset_id, datasetNames)}</dd>

        <dt className="text-slate-500">Evaluate</dt>
        <dd className="text-slate-300">
          {evalIds.length === 0
            ? '—'
            : evalIds.map((id) => name(id, datasetNames)).join(', ')}
        </dd>

        <dt className="text-slate-500">Started</dt>
        <dd className="text-slate-300">{stamp(run.created_at)}</dd>

        <dt className="text-slate-500">{live ? 'Elapsed' : 'Finished'}</dt>
        <dd className="text-slate-300">
          {live ? duration ?? '—' : stamp(run.completed_at)}
        </dd>

        {!live && duration ? (
          <>
            <dt className="text-slate-500">Took</dt>
            <dd className="text-slate-300" data-testid="run-duration">
              {duration}
            </dd>
          </>
        ) : null}
      </dl>

      {children}
    </li>
  );
}
