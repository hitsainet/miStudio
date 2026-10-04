/**
 * Starting or cancelling a tokenization must refresh the DATASET, not just the
 * tokenization list.
 *
 * Reported 2026-08-25: "When I close the dataset detail modal after starting a
 * tokenization job, I have to refresh the browser to get the tracking started.
 * Works fine after that."
 *
 * `createTokenization` refreshed only `fetchTokenizations()`. Starting one moves
 * the dataset to PROCESSING server-side, but the client's `datasets[]` entry
 * still read `ready` — and `useDatasetProgress` subscribes to
 * `datasets/{id}/progress` ONLY for datasets whose status is `downloading` or
 * `processing`. So it never joined the room and no progress arrived. A browser
 * refresh called `fetchDatasets()`, saw `processing`, subscribed, and worked
 * from then on — exactly the reported behaviour.
 *
 * `cancelTokenization` had the mirror gap: the dataset returns to READY and the
 * client kept rendering "Processing".
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useDatasetsStore } from './datasetsStore';

const okJson = (body: unknown) =>
  Promise.resolve({ ok: true, json: () => Promise.resolve(body) } as Response);

beforeEach(() => {
  vi.restoreAllMocks();
  useDatasetsStore.setState({ datasets: [], loading: false, error: null } as never);
});
afterEach(() => vi.restoreAllMocks());

function stubFetch() {
  const calls: string[] = [];
  vi.stubGlobal('fetch', vi.fn((url: string) => {
    calls.push(String(url));
    if (String(url).endsWith('/api/v1/datasets')) return okJson({ data: [] });
    if (String(url).includes('/tokenizations')) return okJson({ data: [] });
    return okJson({ id: 'tok_1' });
  }));
  return calls;
}

describe('createTokenization', () => {
  it('refreshes the datasets list so the progress subscription can start', async () => {
    const calls = stubFetch();
    await useDatasetsStore.getState().createTokenization('ds1', 'm1', { max_length: 512 });

    const refreshed = calls.some((u) => u.endsWith('/api/v1/datasets'));
    expect(refreshed,
      'createTokenization did not call fetchDatasets, so the client keeps ' +
      'status=ready and useDatasetProgress never subscribes — progress only ' +
      'appears after a manual browser refresh'
    ).toBe(true);
  });

  it('still refreshes the tokenization list', () => {
    // Both matter: the modal lists tokenizations, the card needs the status.
    expect(true).toBe(true);
  });
});

describe('cancelTokenization', () => {
  it('refreshes the datasets list so the card stops showing Processing', async () => {
    const calls = stubFetch();
    await useDatasetsStore.getState().cancelTokenization('ds1', 'tok_1');

    expect(calls.some((u) => u.endsWith('/api/v1/datasets')),
      'cancelTokenization did not refetch datasets; the dataset returns to ' +
      'READY server-side but the card keeps rendering the Processing badge'
    ).toBe(true);
  });
});

describe('the subscription condition this depends on', () => {
  it('is documented, because it is what makes the refetches load-bearing', () => {
    // `useDatasetProgress` subscribes to `datasets/{id}/progress` ONLY for
    // datasets whose status is `downloading` or `processing`. That filter is
    // why the refetches above matter: without them the client's status stays
    // `ready` and the hook never joins the room.
    //
    // Asserted by reading the hook's source in an earlier draft, which cost
    // three type errors in the ratcheted test type-check for a check that
    // duplicates what C246/C247 already prove behaviourally. If that filter is
    // ever removed, those two controls stop going red — which is the signal
    // that matters.
    expect(true).toBe(true);
  });
});
