/**
 * Times on the approvals surfaces (ENH-008 §2.7, D8b): the browser's own time zone with its
 * abbreviation as the text, and the exact instant (ISO 8601, UTC) as the tooltip. A missing or
 * unparseable value reads "unknown", never "Invalid Date".
 *
 * ⚠ ECMA-402 REFUSES `timeZoneName` beside `dateStyle`/`timeStyle` (a TypeError at construction),
 * so the build spec's literal options cannot be used; the explicit fields below give the same
 * medium date and short time. Matches miForge's `formatLocal` (ENH-008, 7915b2a). Pinned by
 * `formatLocal.test.ts`, which constructs the real formatter.
 */

export const UNKNOWN_TIME = 'unknown';

export interface LocalTime {
  /** What the element shows. */
  text: string;
  /** The exact instant for `title`; empty when unknown. */
  title: string;
}

function parse(iso: string | null | undefined): Date | null {
  if (iso === null || iso === undefined || iso === '') return null;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatLocal(iso: string | null | undefined, timeZone?: string): LocalTime {
  const date = parse(iso);
  if (date === null) return { text: UNKNOWN_TIME, title: '' };
  const format = new Intl.DateTimeFormat(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    timeZoneName: 'short',
    ...(timeZone === undefined ? {} : { timeZone }),
  });
  return { text: format.format(date), title: date.toISOString() };
}

/** "expires in 23 h" / "expires in 40 min" / "expires in less than 1 min" / "expired" / "never expires". */
export function formatExpiresIn(iso: string | null | undefined, nowMs: number = Date.now()): string {
  if (iso === null || iso === undefined || iso === '') return 'never expires';
  const date = parse(iso);
  if (date === null) return `expires ${UNKNOWN_TIME}`;
  const seconds = Math.round((date.getTime() - nowMs) / 1000);
  if (seconds <= 0) return 'expired';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 1) return 'expires in less than 1 min';
  if (minutes < 60) return `expires in ${minutes} min`;
  return `expires in ${Math.floor(minutes / 60)} h`;
}
