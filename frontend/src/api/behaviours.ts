/**
 * Behaviour catalogue API client (Feature 037, FR-13). Thin: every response is the server's shape.
 */
import { fetchAPI } from './client';
import type {
  Behaviour,
  BehaviourChecklist,
  BehaviourDefinition,
  BehaviourDefinitionBody,
  BehaviourDetail,
  BehaviourHistoryEntry,
  BehaviourList,
  BehaviourStatus,
  BehaviourVersionChange,
  BehaviourWriteResult,
  ProbeRole,
} from '../types/behaviour';

const BASE = '/probe-monitors/behaviours';
const k = (key: string) => `${BASE}/${encodeURIComponent(key)}`;

export const behavioursApi = {
  /** One page of the catalogue (Feature 040 FR-6). Pass the previous page's `next_cursor` for the next. */
  list: (includeArchived = false, page: { cursor?: string | null; limit?: number; q?: string; status?: BehaviourStatus } = {}) => {
    const params = new URLSearchParams({ include_uncatalogued: 'true' });
    if (includeArchived) params.set('include_archived', 'true');
    if (page.limit) params.set('limit', String(page.limit));
    if (page.cursor) params.set('cursor', page.cursor);
    if (page.q) params.set('q', page.q);
    if (page.status) params.set('status', page.status);
    return fetchAPI<BehaviourList>(`${BASE}?${params.toString()}`);
  },
  get: (key: string) => fetchAPI<BehaviourDetail>(k(key)),
  create: (body: BehaviourDefinitionBody & { key: string }) =>
    fetchAPI<BehaviourWriteResult>(BASE, { method: 'POST', body: JSON.stringify(body) }),
  versions: (key: string) => fetchAPI<{ key: string; versions: BehaviourDefinition[] }>(`${k(key)}/versions`),
  /** A full replacement (037), or a change against `base_version` (040 FR-3). */
  newVersion: (key: string, body: BehaviourDefinitionBody | BehaviourVersionChange) =>
    fetchAPI<BehaviourWriteResult>(`${k(key)}/versions`, { method: 'POST', body: JSON.stringify(body) }),
  setRole: (key: string, probeId: string, role: ProbeRole) =>
    fetchAPI<BehaviourDetail>(`${k(key)}/probes/${encodeURIComponent(probeId)}/role`, {
      method: 'PUT',
      body: JSON.stringify({ role }),
    }),
  unlinkView: (key: string, datasetId: string) =>
    fetchAPI<BehaviourDetail>(`${k(key)}/views/${encodeURIComponent(datasetId)}`, { method: 'DELETE' }),
  confirmPositive: (datasetId: string, key: string) =>
    fetchAPI<unknown>(`/probe-monitors/datasets/${encodeURIComponent(datasetId)}/confirm-positive`, {
      method: 'POST',
      body: JSON.stringify({ key }),
    }),
  archive: (key: string) => fetchAPI<Behaviour>(`${k(key)}/archive`, { method: 'POST', body: JSON.stringify({}) }),
  unarchive: (key: string) => fetchAPI<Behaviour>(`${k(key)}/unarchive`, { method: 'POST' }),
  remove: (key: string) => fetchAPI<void>(k(key), { method: 'DELETE' }),
  checklist: (key: string) => fetchAPI<BehaviourChecklist>(`${k(key)}/checklist`),
  history: (key: string) =>
    fetchAPI<{ key: string; total: number; entries: BehaviourHistoryEntry[] }>(`${k(key)}/history?limit=50`),
};
