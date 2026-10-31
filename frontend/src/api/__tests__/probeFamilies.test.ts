/**
 * Feature 041: the family client sends exactly the bodies the routes take (FTASKS 10.3, 14.1). `fetchAPI` is
 * mocked; each method is asserted by URL, method and body.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

const fetchAPI = vi.hoisted(() => vi.fn());
vi.mock('../client', () => ({ fetchAPI, APIError: class extends Error {} }));

import { probeFamiliesApi } from '../probeFamilies';

beforeEach(() => fetchAPI.mockReset().mockResolvedValue({}));

describe('probeFamiliesApi', () => {
  it('marks a suggested blend with exactly the PUT body (no empty reason)', async () => {
    await probeFamiliesApi.setSuggestedBlend('k-x', { model: 'm_1', report_id: 'pbr_1', candidate: 'stack', reason: null });
    await probeFamiliesApi.setSuggestedBlend('k-x', { model: 'm_1', report_id: 'pbr_1', candidate: 'max', reason: 'why' });
    expect(fetchAPI.mock.calls).toEqual([
      ['/probe-monitors/behaviours/k-x/suggested-blend',
        { method: 'PUT', body: JSON.stringify({ model: 'm_1', report_id: 'pbr_1', candidate: 'stack' }) }],
      ['/probe-monitors/behaviours/k-x/suggested-blend',
        { method: 'PUT', body: JSON.stringify({ model: 'm_1', report_id: 'pbr_1', candidate: 'max', reason: 'why' }) }],
    ]);
  });

  it('publishes the family with the model, repository and visibility', async () => {
    await probeFamiliesApi.publishFamily('k-x', { model: 'm_1', repo_id: 'o/r', private: false });
    expect(fetchAPI).toHaveBeenCalledWith('/probe-monitors/behaviours/k-x/family/publish',
      { method: 'POST', body: JSON.stringify({ model: 'm_1', repo_id: 'o/r', private: false }) });
  });

  it('lists reports and prunes rows at their routes', async () => {
    await probeFamiliesApi.listReports('k-x', 'm_1');
    await probeFamiliesApi.pruneReport('pbr_9');
    expect(fetchAPI.mock.calls).toEqual([
      ['/probe-monitors/blend-reports?behaviour_key=k-x&model=m_1'],
      ['/probe-monitors/blend-reports/pbr_9/prune-rows', { method: 'POST', body: '{}' }],
    ]);
  });
});
