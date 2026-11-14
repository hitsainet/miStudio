/**
 * A refetch must not blank out the live-metrics fields.
 *
 * The REST extractions response does not carry status_message, eta_seconds or
 * the sampling metrics — those arrive only over WebSocket. fetchAllExtractions
 * replaced store entries wholesale, so every refetch wiped them.
 *
 * That was harmless while refetching was manual. It stopped being harmless when
 * the 15s reconciliation poll landed (added 2026-07-26 so a dropped
 * `extraction:completed` self-heals): the ETA and status line would now blank
 * on a timer.
 *
 * MUTATION CONTROLS:
 *   * revert to `allExtractions: response.data.data`  -> carry-forward test fails
 *   * carry fields for terminal jobs too              -> terminal test fails
 *   * let carried values win over server values       -> precedence test fails
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';
import { useFeaturesStore } from './featuresStore';

vi.mock('axios');

function respondWith(rows: any[]) {
  (axios.get as any).mockResolvedValue({
    data: { data: rows, meta: { total: rows.length, limit: 50, offset: 0 } },
  });
}

describe('fetchAllExtractions transient-field carry-forward', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useFeaturesStore.setState({ allExtractions: [] });
  });

  it('keeps WebSocket-only fields for a job that is still running', async () => {
    useFeaturesStore.setState({
      allExtractions: [
        {
          id: 'e1',
          status: 'extracting',
          eta_seconds: 900,
          status_message: 'Writing features to database: 10,000/32,768',
          samples_per_second: 12.5,
        } as any,
      ],
    });

    // Server row omits all three, as the REST schema does.
    respondWith([{ id: 'e1', status: 'extracting', progress: 0.92 }]);

    await useFeaturesStore.getState().fetchAllExtractions();

    const row: any = useFeaturesStore.getState().allExtractions[0];
    expect(row.eta_seconds).toBe(900);
    expect(row.status_message).toBe('Writing features to database: 10,000/32,768');
    expect(row.samples_per_second).toBe(12.5);
    // Server state still applied.
    expect(row.progress).toBe(0.92);
  });

  it('drops them once the job reaches a terminal state', async () => {
    useFeaturesStore.setState({
      allExtractions: [
        { id: 'e1', status: 'extracting', eta_seconds: 900 } as any,
      ],
    });

    respondWith([{ id: 'e1', status: 'completed', progress: 1 }]);

    await useFeaturesStore.getState().fetchAllExtractions();

    const row: any = useFeaturesStore.getState().allExtractions[0];
    expect(row.eta_seconds).toBeUndefined();
  });

  it('never lets a carried value override one the server sent', async () => {
    useFeaturesStore.setState({
      allExtractions: [
        { id: 'e1', status: 'extracting', eta_seconds: 900 } as any,
      ],
    });

    respondWith([{ id: 'e1', status: 'extracting', eta_seconds: 42 }]);

    await useFeaturesStore.getState().fetchAllExtractions();

    expect((useFeaturesStore.getState().allExtractions[0] as any).eta_seconds).toBe(42);
  });

  it('leaves a newly appeared job untouched', async () => {
    respondWith([{ id: 'brand_new', status: 'queued' }]);

    await useFeaturesStore.getState().fetchAllExtractions();

    const row: any = useFeaturesStore.getState().allExtractions[0];
    expect(row.id).toBe('brand_new');
    expect(row.eta_seconds).toBeUndefined();
  });
});
