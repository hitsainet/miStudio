/**
 * J-space work in flight, for the model on screen.
 *
 * WHY THIS EXISTS. A 45-minute fit burned the GPU with nothing in the panel
 * saying so. The fit card tracks its own submission in component state, so it
 * only ever knew about a fit THIS browser tab had started — a fit queued from
 * the API, from MCP, from a second tab, or before a refresh was invisible, and
 * the only evidence was the GPU meter in the header.
 *
 * SOURCED FROM `task_queue`, NOT FROM LOCAL STATE. That is the same table the
 * System Monitor's Active Operations reads, so the two cannot disagree about
 * what is running — and a job survives a refresh in the list because the row
 * outlives the page.
 */

import { useEffect, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { getActiveTasks } from '../../api/taskQueue';
import { TaskType, type TaskQueueEntry } from '../../types/taskQueue';

const POLL_MS = 5000;

const LABELS: Partial<Record<TaskType, string>> = {
  [TaskType.JLENS_FIT]: 'Fitting a J-lens',
  [TaskType.JLENS_BAND_REPORT]: 'Measuring the band report',
  [TaskType.JLENS_INTERVENTION]: 'Running an intervention',
  [TaskType.JLENS_READOUT]: 'Reading out',
  [TaskType.JLENS_PROBE]: 'Probing',
  // WITHOUT THESE THE RAW TYPE RENDERS. `LABELS` is a Partial record and the
  // component falls back to `String(task_type)`, so a missing entry shows
  // `jlens_acquire` to the user rather than words.
  [TaskType.JLENS_ACQUIRE]: 'Downloading a published lens',
  [TaskType.JLENS_PUBLISH]: 'Publishing a lens',
};

/**
 * "52.8% · 634/1200" for a fit that has reported counts, else null.
 *
 * Returns NULL rather than a placeholder when the counts are unknown, so the
 * caller falls back to the status. Rendering `0/0` or `— / —` would occupy the
 * space with something that looks like data.
 */
export function counts(row: {
  progress: number | null;
  entity_info?: { prompts_seen?: number; total_prompts?: number } | null;
}): string | null {
  const seen = row.entity_info?.prompts_seen;
  const total = row.entity_info?.total_prompts;
  if (typeof seen !== 'number' || typeof total !== 'number' || total <= 0) {
    return null;
  }
  const pct = row.progress == null ? null : row.progress.toFixed(1);
  const counted = `${seen.toLocaleString()}/${total.toLocaleString()}`;
  return pct == null ? counted : `${pct}% · ${counted}`;
}

/** "1 running, 2 queued" — never a bare total that implies concurrency. */
export function summarise(rows: Array<{ status: string }>): string {
  const running = rows.filter((r) => r.status === 'running').length;
  const queued = rows.filter((r) => r.status === 'queued').length;
  const orphaned = rows.filter((r) => r.status === 'orphaned').length;
  const parts: string[] = [];
  if (running) parts.push(`${running} running`);
  if (queued) parts.push(`${queued} queued`);
  // Surfaced, not hidden: a job whose worker died is the thing a reader most
  // needs to notice, and folding it into a total buries it.
  if (orphaned) parts.push(`${orphaned} stopped reporting`);
  return parts.join(' · ') || 'idle';
}

export function isJSpaceWork(entry: TaskQueueEntry): boolean {
  return String(entry.task_type).startsWith('jlens_');
}

interface RunningWorkProps {
  /** Only work for the model on screen; another model's fit is not this page's. */
  modelId: string;
}

export function RunningWork({ modelId }: RunningWorkProps) {
  const [rows, setRows] = useState<TaskQueueEntry[]>([]);

  useEffect(() => {
    let live = true;
    const tick = async () => {
      try {
        const res = await getActiveTasks();
        if (!live) return;
        setRows((res.data ?? []).filter(isJSpaceWork));
      } catch {
        // A polling failure must not replace a real list with an empty one:
        // "nothing is running" and "I could not ask" look identical, and the
        // first is the reading that stops someone investigating.
      }
    };
    void tick();
    const id = window.setInterval(tick, POLL_MS);
    return () => {
      live = false;
      window.clearInterval(id);
    };
  }, []);

  const mine = rows.filter((r) => !modelId || r.entity_id === modelId);
  if (!mine.length) return null;

  return (
    <section className="mb-4 shrink-0 rounded-lg border border-emerald-300 bg-emerald-50 p-3 dark:border-emerald-700 dark:bg-emerald-950/40">
      <div className="flex flex-wrap items-center gap-2">
        <Loader2
          className={`h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400 ${
            mine.some((r) => r.status === 'running') ? 'animate-spin' : ''
          }`}
        />
        <span className="text-xs font-medium text-emerald-800 dark:text-emerald-300">
          {/* COUNT WHAT IS ACTUALLY RUNNING. This said "3 jobs running" for one
              running job and two queued behind it — a single-GPU queue runs one
              at a time, so the plural was never right. */}
          {summarise(mine)}
        </span>
      </div>
      <ul className="mt-2 space-y-1.5">
        {mine.map((r) => {
          const pct = r.progress == null ? null : Math.round(r.progress);
          return (
            <li key={r.id} className="flex items-center gap-2">
              <span className="w-44 shrink-0 truncate text-[11px] text-emerald-800 dark:text-emerald-300">
                {LABELS[r.task_type] ?? String(r.task_type)}
                {/* WHICH MODEL. Already present in entity_info and simply never
                    read: with two fits queued, a bare percentage cannot say
                    which one is moving. */}
                {r.entity_info?.name ? (
                  <span className="ml-1.5 font-medium">{r.entity_info.name}</span>
                ) : null}
              </span>
              <span className="h-1.5 flex-1 overflow-hidden rounded bg-emerald-200 dark:bg-emerald-900">
                <span
                  className="block h-full bg-emerald-500 transition-all"
                  // A null progress renders as an EMPTY bar, never a full one:
                  // a task that has not reported yet is at the start of its
                  // work, and showing 100% would say the opposite.
                  style={{ width: `${pct ?? 0}%` }}
                />
              </span>
              <span className="w-48 shrink-0 text-right font-mono text-[10px] text-emerald-700 dark:text-emerald-400">
                {counts(r) ?? (pct == null ? r.status : `${pct}% · ${r.status}`)}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
