/**
 * When will this training finish, in the viewer's own timezone?
 *
 * WHY A WINDOWED RATE. Throughput is NOT constant across a run. On
 * `train_c9db4def` it climbed 212 -> 265 -> 290 -> 312 -> 362 steps/min over the
 * first ~1,000 steps while the activation buffer warmed up, then held ~362 for
 * the remaining 59,000. An average taken since `started_at` therefore reads far
 * too pessimistic early on and keeps sliding earlier as the run proceeds, which
 * is the most annoying possible failure for an ETA — it is never wrong in a way
 * you can ignore.
 *
 * The card already keeps the last 20 `(step, timestamp)` pairs for its chart, so
 * a recent-window rate costs no new plumbing. Since-start is kept as the fallback
 * for the first render after a page load, before any WebSocket metric arrives.
 *
 * TIMEZONE. `toLocaleTimeString`/`toLocaleString` format in the host's own
 * timezone by default — there is nothing to configure and nothing to store. The
 * backend stamps UTC, the browser renders local, and the two never need to agree.
 */

/**
 * Statuses with no honest ETA: the run is over, deliberately halted, or has not
 * begun.
 *
 * PAUSED IS THE ONE THAT MATTERS. A paused run keeps its last `current_step` and
 * a recent `updated_at`, so without this it would show a completion time for
 * several minutes — a confident lie about a run that is not moving — until the
 * staleness guard below eventually caught it. The step>0 check already covers
 * pending/initializing, but correctness should not lean on a second guard.
 */
const NOT_PROGRESSING = new Set([
  'completed',
  'failed',
  'cancelled',
  'stopped',
  'paused',
  'pending',
  'initializing',
]);

/**
 * A heartbeat older than this means the run is not currently progressing (paused,
 * stalled, or the worker is gone), so any ETA computed from its rate would be a
 * confident lie that keeps receding. The training loop writes progress far more
 * often than this — `log_interval` 250 at ~362 steps/min is ~41s — so the
 * threshold is generous enough not to flicker on a healthy run.
 */
export const STALE_AFTER_MS = 5 * 60 * 1000;

export interface EtaInputs {
  status: string;
  currentStep: number;
  totalSteps: number;
  /** ISO timestamp; null before the worker picks the job up. */
  startedAt?: string | null;
  /** ISO timestamp of the last row write, for the staleness guard. */
  updatedAt?: string | null;
  /** Recent metric steps, oldest first (the card keeps the last 20). */
  recentSteps?: number[];
  /** Recent metric timestamps, index-aligned with `recentSteps`. */
  recentTimestamps?: string[];
  /** Injected for tests; defaults to now. */
  now?: number;
}

export interface Eta {
  /** Absolute completion instant, epoch ms. */
  at: number;
  /** Milliseconds remaining. */
  remainingMs: number;
  /** Steps per millisecond used, for the tooltip. */
  stepsPerMs: number;
  /** Whether the rate came from the recent window or the whole run. */
  basis: 'recent' | 'since-start';
}

function parse(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const ms = Date.parse(iso);
  return Number.isFinite(ms) ? ms : null;
}

/**
 * Steps per millisecond over the recent window, or null when the window cannot
 * support an estimate (fewer than two points, no forward progress, no elapsed
 * time). Guards against a zero or negative denominator producing Infinity.
 */
export function recentRate(
  steps: number[] | undefined,
  timestamps: string[] | undefined
): number | null {
  if (!steps || !timestamps) return null;
  const n = Math.min(steps.length, timestamps.length);
  if (n < 2) return null;

  const firstStep = steps[0];
  const lastStep = steps[n - 1];
  const firstAt = parse(timestamps[0]);
  const lastAt = parse(timestamps[n - 1]);
  if (firstAt === null || lastAt === null) return null;

  const deltaSteps = lastStep - firstStep;
  const deltaMs = lastAt - firstAt;
  if (deltaSteps <= 0 || deltaMs <= 0) return null;
  return deltaSteps / deltaMs;
}

/**
 * The estimated completion instant, or null when no honest estimate exists.
 *
 * Null — rather than a guess — for: a terminal run, a run that has not started,
 * a run with no forward progress yet, a run already at or past its total, and a
 * run whose heartbeat has gone stale.
 */
export function estimateCompletion(input: EtaInputs): Eta | null {
  const now = input.now ?? Date.now();

  if (NOT_PROGRESSING.has(String(input.status).toLowerCase())) return null;

  const { currentStep, totalSteps } = input;
  if (!Number.isFinite(currentStep) || !Number.isFinite(totalSteps)) return null;
  if (totalSteps <= 0 || currentStep <= 0) return null;
  if (currentStep >= totalSteps) return null;

  // A run whose row has not been written recently is not progressing; an ETA
  // from its stale rate would recede forever.
  const updatedAt = parse(input.updatedAt);
  if (updatedAt !== null && now - updatedAt > STALE_AFTER_MS) return null;

  let stepsPerMs = recentRate(input.recentSteps, input.recentTimestamps);
  let basis: Eta['basis'] = 'recent';

  if (stepsPerMs === null) {
    const startedAt = parse(input.startedAt);
    if (startedAt === null) return null;
    const elapsed = now - startedAt;
    if (elapsed <= 0) return null;
    stepsPerMs = currentStep / elapsed;
    basis = 'since-start';
  }

  if (!Number.isFinite(stepsPerMs) || stepsPerMs <= 0) return null;

  const remainingMs = (totalSteps - currentStep) / stepsPerMs;
  if (!Number.isFinite(remainingMs) || remainingMs < 0) return null;

  return { at: now + remainingMs, remainingMs, stepsPerMs, basis };
}

/** "2h 45m", "45m", "30s" — the coarsest useful unit, never "0m". */
export function formatRemaining(ms: number): string {
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m`;
  return `${seconds}s`;
}

/**
 * The clock time to show, in the VIEWER'S timezone. Same-day runs show a time
 * only; one crossing midnight shows the date too, because "2:30 AM" with no date
 * is ambiguous on a 2h45m run started at 11:44 PM — which is exactly what these
 * runs do.
 */
export function formatEtaClock(at: number, now: number = Date.now()): string {
  const eta = new Date(at);
  const sameDay = new Date(now).toDateString() === eta.toDateString();
  return sameDay
    ? eta.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
    : eta.toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
      });
}

/** The card's tooltip: how the estimate was made, so it can be judged. */
export function etaTooltip(eta: Eta): string {
  const perMin = eta.stepsPerMs * 60_000;
  const basis =
    eta.basis === 'recent'
      ? 'from the recent step rate'
      : 'from the average rate since the run started (no recent metrics yet)';
  return `About ${formatRemaining(eta.remainingMs)} remaining, ${basis}: ${perMin.toFixed(
    0
  )} steps/min. Shown in your local timezone.`;
}
