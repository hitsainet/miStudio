/**
 * An older feature-list response must not overwrite a newer request's state.
 *
 * Opening the feature browser fetches with the default filters; turning on the starred filter
 * fetches again with `is_favorite: true`. When the OPENING response landed last, it wrote its own
 * filters back over the starred ones, so the starred filter switched itself off (and the batch
 * enhanced-labeling button it gates vanished). React 19's render timing made that ordering the
 * common one in ExtractionJobCard.queue.test.tsx; on a real network either order can happen.
 *
 * MUTATION CONTROLS:
 *   * remove `if (!isLatest()) return;` after the await (either fetch) -> the matching test fails
 *   * remove it from the catch                                          -> the error test fails
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';
import { useFeaturesStore } from './featuresStore';
import type { FeatureSearchRequest } from '../types/features';

vi.mock('axios');

type Deferred = { resolve: (v: unknown) => void; reject: (e: unknown) => void };

function deferredGets(): Deferred[] {
  const pending: Deferred[] = [];
  vi.mocked(axios.get).mockImplementation(
    () => new Promise((resolve, reject) => pending.push({ resolve, reject })),
  );
  return pending;
}

const DEFAULTS: FeatureSearchRequest = { limit: 50, offset: 0 };
const STARRED: FeatureSearchRequest = { limit: 50, offset: 0, is_favorite: true };

function rowIds(rowsKey: 'featuresByExtraction' | 'featuresByTraining'): string[] {
  return (useFeaturesStore.getState()[rowsKey].x1 ?? []).map((f) => f.id);
}

const page = (ids: number[]) => ({
  data: { features: ids.map((id) => ({ id: `f${id}` })), total: ids.length, limit: 50, offset: 0 },
});

describe.each([
  ['fetchExtractionFeatures', 'featuresByExtraction'],
  ['fetchFeatures', 'featuresByTraining'],
] as const)('%s with out-of-order responses', (action, rowsKey) => {
  beforeEach(() => {
    vi.clearAllMocks();
    useFeaturesStore.setState({ searchFilters: {}, featuresByExtraction: {}, featuresByTraining: {}, featuresError: null });
  });

  it('keeps the newer request\'s filters and rows when the older response lands last', async () => {
    const pending = deferredGets();
    const fetch = useFeaturesStore.getState()[action];
    const older = fetch('x1', DEFAULTS);
    const newer = fetch('x1', STARRED);
    expect(pending).toHaveLength(2);

    pending[1].resolve(page([2]));
    await newer;
    pending[0].resolve(page([1, 1]));
    await older;

    const s = useFeaturesStore.getState();
    expect(s.searchFilters.x1).toMatchObject({ is_favorite: true });
    expect(rowIds(rowsKey)).toEqual(['f2']);
    expect(s.isLoadingFeatures).toBe(false);
  });

  it('does not report an older request\'s failure over a newer success', async () => {
    const pending = deferredGets();
    const fetch = useFeaturesStore.getState()[action];
    const older = fetch('x1', DEFAULTS);
    const newer = fetch('x1', STARRED);

    pending[1].resolve(page([2]));
    await newer;
    pending[0].reject(new Error('network down'));
    await older;

    expect(useFeaturesStore.getState().featuresError).toBeNull();
  });

  it('still applies a response when it is the latest request', async () => {
    const pending = deferredGets();
    const done = useFeaturesStore.getState()[action]('x1', DEFAULTS);
    pending[0].resolve(page([7]));
    await done;
    expect(rowIds(rowsKey)).toEqual(['f7']);
  });
});
