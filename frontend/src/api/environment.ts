/**
 * Runtime environment settings (Feature 036; ENH-003 §3.1, D3).
 *
 * `GET /api/v1/environment` lists every setting the Settings → Environment tab shows;
 * `PUT /environment/{key}` stores an operator override, `DELETE` returns the setting to the
 * environment's value, `GET /environment/history` lists the newest changes.
 */
import { fetchAPI } from './client';

export type EnvironmentGroup = 'runtime' | 'restart' | 'environment';

export interface EnvironmentLastChange {
  actor: string;
  action: 'set' | 'reset';
  at: string;
}

export interface EnvironmentRow {
  label: string;
  env_name: string;
  description: string;
  group: EnvironmentGroup;
  /** Operator-facing category id (one of `EnvironmentListing.categories`); the tab groups by it. */
  category: string;
  value: unknown;
  env_value: unknown;
  source: 'env' | 'override' | 'external';
  runtime_writable: boolean;
  effect: string | null;
  processes: string[];
  restart: string | null;
  restart_warning: string | null;
  secret: boolean;
  is_set: boolean | null;
  updated_at: string | null;
  updated_by: string | null;
  reason: string | null;
  last_change: EnvironmentLastChange | null;
}

export interface EnvironmentCategory {
  id: string;
  label: string;
}

export interface EnvironmentListing {
  settings: Record<string, EnvironmentRow>;
  /** Every category, in the order the page shows them. */
  categories: EnvironmentCategory[];
  count: number;
  runtime_keys: string[];
  overrides: Record<string, unknown>;
}

export interface EnvironmentWriteResult {
  key: string;
  value: unknown;
  previous: unknown;
  source: 'env' | 'override';
}

export interface EnvironmentChange {
  id: number;
  key: string;
  action: 'set' | 'reset';
  previous_value: unknown;
  previous_source: string;
  new_value: unknown;
  new_source: string;
  actor: string;
  reason: string | null;
  changed_at: string;
}

export interface EnvironmentHistory {
  changes: EnvironmentChange[];
  count: number;
}

export function readEnvironment(): Promise<EnvironmentListing> {
  return fetchAPI<EnvironmentListing>('/environment');
}

export function readEnvironmentHistory(limit = 50): Promise<EnvironmentHistory> {
  return fetchAPI<EnvironmentHistory>(`/environment/history?limit=${limit}`);
}

export function writeEnvironment(
  key: string,
  value: unknown,
  reason?: string,
): Promise<EnvironmentWriteResult> {
  const body: { value: unknown; reason?: string } = { value };
  if (reason && reason.trim()) body.reason = reason.trim();
  return fetchAPI<EnvironmentWriteResult>(`/environment/${encodeURIComponent(key)}`, {
    method: 'PUT',
    body: JSON.stringify(body),
  });
}

export function resetEnvironment(key: string, reason?: string): Promise<EnvironmentWriteResult> {
  const query = reason && reason.trim() ? `?reason=${encodeURIComponent(reason.trim())}` : '';
  return fetchAPI<EnvironmentWriteResult>(`/environment/${encodeURIComponent(key)}${query}`, {
    method: 'DELETE',
  });
}
