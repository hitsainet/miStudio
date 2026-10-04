/**
 * A background refresh must not blank the list that is already on screen.
 *
 * REPORTED 2026-07-26: "the whole page refreshes to update progress on the
 * extraction page and closes the expanded progress windows when it does ...
 * 15 secs is probably correct."
 *
 * 15s is exactly the reconciliation poll added earlier that day. The chain:
 *
 *   poll -> fetchAllExtractions
 *        -> set({ isLoadingExtractions: true })
 *        -> ExtractionsPanel renders the grid behind !isLoadingExtractions,
 *           so every ExtractionJobCard UNMOUNTS
 *        -> card-local showMetrics resets, expanded panels snap shut
 *
 * Fixed in the store rather than in the poll: all four callers trigger it
 * (the poll, extraction:completed, extraction:failed, nlp completion), and a
 * refresh of a visible list should never blank it regardless of who asked.
 *
 * MUTATION CONTROLS:
 *   * restore the unconditional `isLoadingExtractions: true` -> refresh test fails
 *   * never set it, even on first load                       -> first-load test fails
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';
import { useFeaturesStore } from './featuresStore';

vi.mock('axios');

describe('fetchAllExtractions loading flag', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useFeaturesStore.setState({ allExtractions: [], isLoadingExtractions: false });
  });

  it('does not flip the loading flag when a list is already displayed', async () => {
    useFeaturesStore.setState({
      allExtractions: [{ id: 'e1', status: 'extracting' } as any],
    });

    const seen: boolean[] = [];
    const unsub = useFeaturesStore.subscribe((state) =>
      seen.push(state.isLoadingExtractions),
    );

    (axios.get as any).mockResolvedValue({
      data: {
        data: [{ id: 'e1', status: 'extracting', progress: 0.5 }],
        meta: { total: 1, limit: 50, offset: 0 },
      },
    });

    await useFeaturesStore.getState().fetchAllExtractions();
    unsub();

    expect(seen).not.toContain(true);
  });

  it('still shows a spinner on the very first load', async () => {
    const seen: boolean[] = [];
    const unsub = useFeaturesStore.subscribe((state) =>
      seen.push(state.isLoadingExtractions),
    );

    (axios.get as any).mockResolvedValue({
      data: {
        data: [{ id: 'e1', status: 'queued' }],
        meta: { total: 1, limit: 50, offset: 0 },
      },
    });

    await useFeaturesStore.getState().fetchAllExtractions();
    unsub();

    expect(seen).toContain(true);
    expect(useFeaturesStore.getState().isLoadingExtractions).toBe(false);
  });

  it('clears the flag even when the refresh fails', async () => {
    (axios.get as any).mockRejectedValue(new Error('boom'));

    await useFeaturesStore.getState().fetchAllExtractions();

    expect(useFeaturesStore.getState().isLoadingExtractions).toBe(false);
    expect(useFeaturesStore.getState().extractionsError).toBeTruthy();
  });
});
