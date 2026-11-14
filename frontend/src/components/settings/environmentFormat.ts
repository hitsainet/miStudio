/**
 * Value formatting and draft parsing for Settings → Environment (Feature 036).
 * Kept apart from the component so fast refresh works (react-refresh/only-export-components).
 */

/** `core.config.Settings.log_level`'s `Literal`. The backend refuses anything else. */
export const LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] as const;

export type Editor = 'boolean' | 'log-level' | 'number' | 'json' | 'text';

export function editorFor(key: string, value: unknown): Editor {
  if (key === 'log_level') return 'log-level';
  if (key.endsWith('_json')) return 'json';
  if (typeof value === 'boolean') return 'boolean';
  if (typeof value === 'number') return 'number';
  return 'text';
}

/** A value as the operator reads it. `null` is "not set", never an empty cell. */
export function showValue(value: unknown): string {
  if (value === null || value === undefined) return 'not set';
  if (typeof value === 'boolean') return value ? 'on' : 'off';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

export function draftOf(value: unknown): string {
  return value === null || value === undefined ? '' : String(value);
}

/** The typed draft as the value `PUT` sends, or why it cannot be one. */
export function parseDraft(editor: Editor, draft: string): { value: unknown } | { error: string } {
  const text = draft.trim();
  switch (editor) {
    case 'boolean':
      return text === 'true' || text === 'false'
        ? { value: text === 'true' }
        : { error: 'Choose on or off' };
    case 'number': {
      const number = Number(text);
      return text === '' || !Number.isFinite(number) ? { error: 'Enter a number' } : { value: number };
    }
    case 'log-level':
      return { value: text };
    case 'json':
      return { value: text === '' ? '{}' : text };
    case 'text':
      return { value: text === '' ? null : text };
  }
}

/** One category's rows, as a section of the Environment tab shows them. */
export interface CategoryGroup<R> {
  id: string;
  label: string;
  rows: [string, R][];
}

/**
 * Rows grouped by operator-facing category, in the backend's category order, each group sorted
 * by label (case-insensitive, then by key so the order never depends on the payload's). Empty
 * categories are left out; a row naming a category the listing does not list goes in a trailing
 * "Other" group rather than disappearing.
 */
export function groupByCategory<R extends { label: string; category?: string | null }>(
  rows: [string, R][],
  categories: { id: string; label: string }[] | undefined,
): CategoryGroup<R>[] {
  const order = categories ?? [];
  const known = new Set(order.map((c) => c.id));
  const byLabel = (a: [string, R], b: [string, R]) =>
    a[1].label.localeCompare(b[1].label, undefined, { sensitivity: 'base' }) || a[0].localeCompare(b[0]);
  const groups: CategoryGroup<R>[] = order.map((c) => ({
    id: c.id,
    label: c.label,
    rows: rows.filter(([, r]) => r.category === c.id),
  }));
  const stray = rows.filter(([, r]) => !r.category || !known.has(r.category));
  if (stray.length > 0) {
    const other = groups.find((g) => g.id === 'other');
    if (other) other.rows.push(...stray);
    else groups.push({ id: 'other', label: 'Other', rows: stray });
  }
  return groups
    .filter((g) => g.rows.length > 0)
    .map((g) => ({ ...g, rows: [...g.rows].sort(byLabel) }));
}
