/**
 * The probe-dataset archive/purge/rows calls go to the routes the backend serves, with the
 * payloads it reads (2026-10-21). Asserted on the wire (`fetch`), not on the store.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { probeMonitorsApi } from './probeMonitors';

const fetchMock = vi.fn();
globalThis.fetch = fetchMock as unknown as typeof fetch;

function ok(body: unknown) {
  return { ok: true, status: 200, json: async () => body };
}

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(ok({ items: [], results: [], purged: 0, blocked: 0 }));
});

function call(n = 0): { url: string; method: string; body: unknown } {
  const [url, init] = fetchMock.mock.calls[n];
  return { url, method: init?.method ?? 'GET', body: init?.body ? JSON.parse(init.body) : undefined };
}

describe('probe dataset archive, purge and rows calls', () => {
  it('lists WITH archived views, so run tiles can still name them', async () => {
    await probeMonitorsApi.listDatasets('eval');
    expect(call().url).toMatch(/\/probe-monitors\/datasets\?include_archived=true&role=eval$/);
  });

  it('archives and unarchives by id', async () => {
    await probeMonitorsApi.archiveDataset('pmd_1', 'superseded');
    expect(call(0)).toMatchObject({ method: 'POST', body: { reason: 'superseded' } });
    expect(call(0).url).toMatch(/\/probe-monitors\/datasets\/pmd_1\/archive$/);
    await probeMonitorsApi.unarchiveDataset('pmd_1');
    expect(call(1).url).toMatch(/\/probe-monitors\/datasets\/pmd_1\/unarchive$/);
    expect(call(1).method).toBe('POST');
  });

  it('plans and purges exactly the ids given', async () => {
    await probeMonitorsApi.planDatasetPurge(['pmd_1', 'pmd_2']);
    expect(call(0).url).toMatch(/\/probe-monitors\/datasets\/purge\/plan$/);
    expect(call(0).body).toEqual({ ids: ['pmd_1', 'pmd_2'] });
    await probeMonitorsApi.purgeDatasets(['pmd_2'], 'audit');
    expect(call(1).url).toMatch(/\/probe-monitors\/datasets\/purge$/);
    expect(call(1)).toMatchObject({ method: 'POST', body: { ids: ['pmd_2'], reason: 'audit' } });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('asks for rows with every control as a query parameter', async () => {
    await probeMonitorsApi.datasetRows('pmd_1', {
      offset: 100, limit: 25, columns: ['source', 'notes'], label: 'excluded', search: 'card',
    });
    const url = new URL(call().url, 'http://x');
    expect(url.pathname).toMatch(/\/probe-monitors\/datasets\/pmd_1\/rows$/);
    expect(Object.fromEntries(url.searchParams)).toEqual({
      offset: '100', limit: '25', columns: 'source,notes', label: 'excluded', search: 'card',
    });
  });

  it('an empty projection asks for no other columns; no projection asks for all', async () => {
    await probeMonitorsApi.datasetRows('pmd_1', { columns: [] });
    expect(new URL(call(0).url, 'http://x').searchParams.get('columns')).toBe('');
    await probeMonitorsApi.datasetRows('pmd_1', {});
    expect(new URL(call(1).url, 'http://x').searchParams.has('columns')).toBe(false);
  });
});
