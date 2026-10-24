/** Shared formatting for the probe dataset row browser (kept out of the component file so
 * React fast refresh keeps working). */
import type { ProbeRowOutcome } from '../../types/probeMonitor';

export const PAGE_SIZES = [25, 50, 100, 200] as const;

/** Plain words for the five outcomes, in the order the counts are shown. */
export const OUTCOME_LABELS: Record<ProbeRowOutcome, string> = {
  positive: 'Positive',
  negative: 'Negative',
  excluded: 'Excluded',
  filtered_out: 'Filtered out',
  unparseable: 'Unreadable',
};

export const PREVIEW_CHARS = 160;

/** A cell as text: a conversation as "role: content" lines, anything else as JSON. */
export function cellText(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  if (Array.isArray(value) && value.every((m) => m && typeof m === 'object' && 'content' in m)) {
    return value
      .map((m) => `${(m as { role?: string }).role ?? '?'}: ${(m as { content: unknown }).content}`)
      .join('\n');
  }
  return JSON.stringify(value);
}

